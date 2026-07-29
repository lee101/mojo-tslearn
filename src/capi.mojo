"""Time-series distance and alignment kernels exposed through a C ABI."""

from std.algorithm import sync_parallelize
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.math import exp, log, sqrt
from std.sys.info import simd_width_of

comptime W = simd_width_of[DType.float64]()
comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime INF = 1.7976931348623157e308


@always_inline
def sqeuclidean(a: Ptr, b: Ptr, d: Int) -> Float64:
    if d == 1:
        var delta = a[0] - b[0]
        return delta * delta
    if d == 2:
        var delta0 = a[0] - b[0]
        var delta1 = a[1] - b[1]
        return delta0 * delta0 + delta1 * delta1
    if d == 3:
        var delta0 = a[0] - b[0]
        var delta1 = a[1] - b[1]
        var delta2 = a[2] - b[2]
        return delta0 * delta0 + delta1 * delta1 + delta2 * delta2
    var lanes = SIMD[DType.float64, W](0.0)
    var k = 0
    while k + W <= d:
        var delta = a.load[width=W](k) - b.load[width=W](k)
        lanes += delta * delta
        k += W
    var total = lanes.reduce_add()
    while k < d:
        var delta = a[k] - b[k]
        total += delta * delta
        k += 1
    return total


def dtw_fill(
    s1: Ptr,
    s2: Ptr,
    n: Int,
    m: Int,
    d: Int,
    mask: Ptr,
    use_mask: Bool,
    acc: Ptr,
) -> Float64:
    var stride = m + 1
    if use_mask:
        for idx in range((n + 1) * stride):
            acc[idx] = INF
    else:
        for j in range(stride):
            acc[j] = INF
        for i in range(1, n + 1):
            acc[i * stride] = INF
    acc[0] = 0.0
    for i in range(n):
        for j in range(m):
            if use_mask and mask[i * m + j] == 0.0:
                continue
            var left = acc[(i + 1) * stride + j]
            var up = acc[i * stride + j + 1]
            var diag = acc[i * stride + j]
            var best = diag
            if up < best:
                best = up
            if left < best:
                best = left
            acc[(i + 1) * stride + j + 1] = (
                sqeuclidean(s1 + i * d, s2 + j * d, d) + best
            )
    return sqrt(acc[n * stride + m])


def dtw_backtrack(acc: Ptr, n: Int, m: Int, path_i: IPtr, path_j: IPtr) -> Int:
    var i = n - 1
    var j = m - 1
    var length = 0
    var stride = m + 1
    while True:
        path_i[length] = Int64(i)
        path_j[length] = Int64(j)
        length += 1
        if i == 0 and j == 0:
            break
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            var diag = acc[i * stride + j]
            var up = acc[i * stride + j + 1]
            var left = acc[(i + 1) * stride + j]
            if diag <= up and diag <= left:
                i -= 1
                j -= 1
            elif up <= left:
                i -= 1
            else:
                j -= 1
    return length


@always_inline
def softmin3(
    a: Float64, b: Float64, c: Float64, gamma: Float64, inv_gamma: Float64
) -> Float64:
    var pivot = a
    if b < pivot:
        pivot = b
    if c < pivot:
        pivot = c
    if pivot == a:
        return pivot - gamma * log(
            1.0
            + exp((pivot - b) * inv_gamma)
            + exp((pivot - c) * inv_gamma)
        )
    if pivot == b:
        return pivot - gamma * log(
            exp((pivot - a) * inv_gamma)
            + 1.0
            + exp((pivot - c) * inv_gamma)
        )
    return pivot - gamma * log(
        exp((pivot - a) * inv_gamma)
        + exp((pivot - b) * inv_gamma)
        + 1.0
    )


def soft_dtw_fill(
    s1: Ptr, s2: Ptr, n: Int, m: Int, d: Int, gamma: Float64, acc: Ptr
) -> Float64:
    if gamma == 0.0:
        var dist = dtw_fill(s1, s2, n, m, d, acc, False, acc)
        return dist * dist
    var stride = m + 1
    for j in range(stride):
        acc[j] = INF
    acc[0] = 0.0
    var inv_gamma = 1.0 / gamma
    for i in range(1, n + 1):
        var previous = acc + ((i - 1) % 2) * stride
        var current = acc + (i % 2) * stride
        current[0] = INF
        for j in range(1, m + 1):
            current[j] = (
                sqeuclidean(s1 + (i - 1) * d, s2 + (j - 1) * d, d)
                + softmin3(
                    previous[j],
                    current[j - 1],
                    previous[j - 1],
                    gamma,
                    inv_gamma,
                )
            )
    return acc[(n % 2) * stride + m]


def soft_dtw_gpu_kernel(
    x: Ptr,
    y: Ptr,
    nx: Int,
    ny: Int,
    sx: Int,
    sy: Int,
    d: Int,
    gamma: Float64,
    result: Ptr,
    scratch: Ptr,
):
    var pair = block_idx.x * block_dim.x + thread_idx.x
    if pair >= nx * ny:
        return
    var i = pair // ny
    var j = pair % ny
    var stride = sy + 1
    var acc = scratch + pair * 2 * stride
    for column in range(stride):
        acc[column] = INF
    acc[0] = 0.0
    var inv_gamma = 1.0 / gamma
    for row in range(1, sx + 1):
        var previous = acc + ((row - 1) % 2) * stride
        var current = acc + (row % 2) * stride
        current[0] = INF
        for column in range(1, sy + 1):
            current[column] = (
                sqeuclidean(
                    x + (i * sx + row - 1) * d,
                    y + (j * sy + column - 1) * d,
                    d,
                )
                + softmin3(
                    previous[column],
                    current[column - 1],
                    previous[column - 1],
                    gamma,
                    inv_gamma,
                )
            )
    result[pair] = acc[(sx % 2) * stride + sy]


def cost_dtw_fill(
    cost: Ptr, n: Int, m: Int, mask: Ptr, use_mask: Bool, acc: Ptr
) -> Float64:
    var stride = m + 1
    if use_mask:
        for idx in range((n + 1) * stride):
            acc[idx] = INF
    else:
        for j in range(stride):
            acc[j] = INF
        for i in range(1, n + 1):
            acc[i * stride] = INF
    acc[0] = 0.0
    for i in range(n):
        for j in range(m):
            if use_mask and mask[i * m + j] == 0.0:
                continue
            var left = acc[(i + 1) * stride + j]
            var up = acc[i * stride + j + 1]
            var diag = acc[i * stride + j]
            var best = diag
            if up < best:
                best = up
            if left < best:
                best = left
            acc[(i + 1) * stride + j + 1] = cost[i * m + j] + best
    return acc[n * stride + m]


def dba_accumulate(
    center: Ptr,
    series: Ptr,
    n: Int,
    m: Int,
    d: Int,
    weight: Float64,
    sums: Ptr,
    counts: Ptr,
    acc: Ptr,
) -> Float64:
    var dist = dtw_fill(center, series, n, m, d, acc, False, acc)
    var i = n - 1
    var j = m - 1
    var stride = m + 1
    while True:
        counts[i] += weight
        for k in range(d):
            sums[i * d + k] += weight * series[j * d + k]
        if i == 0 and j == 0:
            break
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            var diag = acc[i * stride + j]
            var up = acc[i * stride + j + 1]
            var left = acc[(i + 1) * stride + j]
            if diag <= up and diag <= left:
                i -= 1
                j -= 1
            elif up <= left:
                i -= 1
            else:
                j -= 1
    return dist * dist * weight


@export("mts_dtw")
def mts_dtw(
    s1_addr: Int,
    s2_addr: Int,
    n: Int,
    m: Int,
    d: Int,
    mask_addr: Int,
    use_mask: Int,
    acc_addr: Int,
) abi("C") -> Float64:
    return dtw_fill(
        Ptr(unsafe_from_address=s1_addr),
        Ptr(unsafe_from_address=s2_addr),
        n,
        m,
        d,
        Ptr(unsafe_from_address=mask_addr),
        use_mask != 0,
        Ptr(unsafe_from_address=acc_addr),
    )


@export("mts_dtw_backtrack")
def mts_dtw_backtrack(
    acc_addr: Int, n: Int, m: Int, path_i_addr: Int, path_j_addr: Int
) abi("C") -> Int:
    return dtw_backtrack(
        Ptr(unsafe_from_address=acc_addr),
        n,
        m,
        IPtr(unsafe_from_address=path_i_addr),
        IPtr(unsafe_from_address=path_j_addr),
    )


@export("mts_soft_dtw")
def mts_soft_dtw(
    s1_addr: Int,
    s2_addr: Int,
    n: Int,
    m: Int,
    d: Int,
    gamma: Float64,
    acc_addr: Int,
) abi("C") -> Float64:
    return soft_dtw_fill(
        Ptr(unsafe_from_address=s1_addr),
        Ptr(unsafe_from_address=s2_addr),
        n,
        m,
        d,
        gamma,
        Ptr(unsafe_from_address=acc_addr),
    )


@export("mts_dtw_from_cost")
def mts_dtw_from_cost(
    cost_addr: Int,
    n: Int,
    m: Int,
    mask_addr: Int,
    use_mask: Int,
    acc_addr: Int,
) abi("C") -> Float64:
    return cost_dtw_fill(
        Ptr(unsafe_from_address=cost_addr),
        n,
        m,
        Ptr(unsafe_from_address=mask_addr),
        use_mask != 0,
        Ptr(unsafe_from_address=acc_addr),
    )


@export("mts_cdist_dtw")
def mts_cdist_dtw(
    x_addr: Int,
    y_addr: Int,
    nx: Int,
    ny: Int,
    sx: Int,
    sy: Int,
    d: Int,
    result_addr: Int,
    acc_addr: Int,
    parallel: Int,
) abi("C"):
    var x = Ptr(unsafe_from_address=x_addr)
    var y = Ptr(unsafe_from_address=y_addr)
    var result = Ptr(unsafe_from_address=result_addr)
    var acc = Ptr(unsafe_from_address=acc_addr)
    var symmetric = x_addr == y_addr and nx == ny and sx == sy
    var acc_stride = (sx + 1) * (sy + 1)

    @parameter
    def compute_row(i: Int):
        var row_acc = acc
        if parallel != 0:
            row_acc += i * acc_stride
        var start = 0
        if symmetric:
            result[i * ny + i] = 0.0
            start = i + 1
        for j in range(start, ny):
            var distance = dtw_fill(
                x + i * sx * d,
                y + j * sy * d,
                sx,
                sy,
                d,
                row_acc,
                False,
                row_acc,
            )
            result[i * ny + j] = distance
            if symmetric:
                result[j * ny + i] = distance

    if parallel != 0:
        sync_parallelize[compute_row](nx)
    else:
        for i in range(nx):
            compute_row(i)


@export("mts_cdist_soft_dtw")
def mts_cdist_soft_dtw(
    x_addr: Int,
    y_addr: Int,
    nx: Int,
    ny: Int,
    sx: Int,
    sy: Int,
    d: Int,
    gamma: Float64,
    result_addr: Int,
    acc_addr: Int,
) abi("C"):
    var x = Ptr(unsafe_from_address=x_addr)
    var y = Ptr(unsafe_from_address=y_addr)
    var result = Ptr(unsafe_from_address=result_addr)
    var acc = Ptr(unsafe_from_address=acc_addr)
    for i in range(nx):
        for j in range(ny):
            result[i * ny + j] = soft_dtw_fill(
                x + i * sx * d,
                y + j * sy * d,
                sx,
                sy,
                d,
                gamma,
                acc,
            )


@export("mts_cdist_soft_dtw_gpu")
def mts_cdist_soft_dtw_gpu(
    x_addr: Int,
    y_addr: Int,
    nx: Int,
    ny: Int,
    sx: Int,
    sy: Int,
    d: Int,
    gamma: Float64,
    result_addr: Int,
) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        var x_size = nx * sx * d
        var y_size = ny * sy * d
        var result_size = nx * ny
        var scratch_size = result_size * 2 * (sy + 1)
        var x_device = ctx.enqueue_create_buffer[DType.float64](x_size)
        var y_device = ctx.enqueue_create_buffer[DType.float64](y_size)
        var result_device = ctx.enqueue_create_buffer[DType.float64](result_size)
        var scratch_device = ctx.enqueue_create_buffer[DType.float64](scratch_size)
        ctx.enqueue_copy(x_device, Ptr(unsafe_from_address=x_addr))
        ctx.enqueue_copy(y_device, Ptr(unsafe_from_address=y_addr))
        comptime block_size = 128
        ctx.enqueue_function[soft_dtw_gpu_kernel](
            x_device,
            y_device,
            nx,
            ny,
            sx,
            sy,
            d,
            gamma,
            result_device,
            scratch_device,
            grid_dim=(result_size + block_size - 1) // block_size,
            block_dim=block_size,
        )
        ctx.enqueue_copy(
            Ptr(unsafe_from_address=result_addr),
            result_device,
        )
        ctx.synchronize()
        return 1
    except:
        return 0


@export("mts_cdist_euclidean")
def mts_cdist_euclidean(
    x_addr: Int,
    y_addr: Int,
    nx: Int,
    ny: Int,
    width: Int,
    result_addr: Int,
) abi("C"):
    var x = Ptr(unsafe_from_address=x_addr)
    var y = Ptr(unsafe_from_address=y_addr)
    var result = Ptr(unsafe_from_address=result_addr)
    for i in range(nx):
        for j in range(ny):
            result[i * ny + j] = sqrt(
                sqeuclidean(x + i * width, y + j * width, width)
            )


@export("mts_dba_accumulate")
def mts_dba_accumulate(
    center_addr: Int,
    series_addr: Int,
    n: Int,
    m: Int,
    d: Int,
    weight: Float64,
    sums_addr: Int,
    counts_addr: Int,
    acc_addr: Int,
) abi("C") -> Float64:
    return dba_accumulate(
        Ptr(unsafe_from_address=center_addr),
        Ptr(unsafe_from_address=series_addr),
        n,
        m,
        d,
        weight,
        Ptr(unsafe_from_address=sums_addr),
        Ptr(unsafe_from_address=counts_addr),
        Ptr(unsafe_from_address=acc_addr),
    )
