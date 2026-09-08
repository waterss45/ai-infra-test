// 朴素 float 矩阵乘 kernel（被测对象，供 ctypes 加载）
// 构建: scripts/build_native.sh (Linux/g++) 或 CI
#include <cstddef>

extern "C" {

void mm_f32(const float* a, const float* b, float* out, size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
}

int add_one(int x) { return x + 1; }

}
