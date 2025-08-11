// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  func.func @test_scatter(%arg0: tensor<71x32xbf16>, %arg1: tensor<71x4x2xi64>, %arg2: tensor<71x4xbf16>) -> tensor<71x32xbf16> {
    // CHECK: %0 = tensor.empty() : tensor<71x32xbf16>
    %0 = ttir.empty() : tensor<71x32xbf16>
    // CHECK: %1 = tosa.const_shape  {values = dense<[1, 71, 32]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK: %2 = tosa.reshape %arg0, %1 : (tensor<71x32xbf16>, !tosa.shape<3>) -> tensor<1x71x32xbf16>
    // CHECK: %3 = tosa.const_shape  {values = dense<[1, 71, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    // CHECK: %4 = tosa.reshape %arg2, %3 : (tensor<71x4xbf16>, !tosa.shape<3>) -> tensor<1x71x4xbf16>
    // CHECK: %5 = tosa.const_shape  {values = dense<[284, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK: %6 = tosa.reshape %arg1, %5 : (tensor<71x4x2xi64>, !tosa.shape<2>) -> tensor<284x2xi64>
    // CHECK: %7 = tosa.cast %6 : (tensor<284x2xi64>) -> tensor<284x2xi32>
    // CHECK: %8 = tosa.scatter %4, %7, %2 : (tensor<1x71x4xbf16>, tensor<284x2xi32>, tensor<1x71x32xbf16>) -> tensor<1x71x32xbf16>
    // CHECK: %9 = tosa.const_shape  {values = dense<[71, 32]> : tensor<2xindex>} : () -> !tosa.shape<2>
    // CHECK: %10 = tosa.reshape %8, %9 : (tensor<1x71x32xbf16>, !tosa.shape<2>) -> tensor<71x32xbf16>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0, 1>, scatter_dims_to_operand_dims = array<i32: 0, 1>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32>}> : (tensor<71x32xbf16>, tensor<71x4x2xi64>, tensor<71x4xbf16>, tensor<71x32xbf16>) -> tensor<71x32xbf16>
    // CHECK: return %10 : tensor<71x32xbf16>
    return %1 : tensor<71x32xbf16>
  }
}
