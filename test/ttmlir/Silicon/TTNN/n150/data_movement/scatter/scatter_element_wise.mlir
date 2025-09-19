// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// The following are simple element-wise scatter tests
// shape(input) == shape(output)
// single index dimension (len(scatter_dims_to_operand_dims) == 1)

func.func @scatter_simple_1(%arg0: tensor<1x3x320x320xf32>, %arg1: tensor<1x1xi32>, %arg2: tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32> {
  %0 = ttir.empty() : tensor<1x3x320x320xf32>
  %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 1, 2, 3>}> : (tensor<1x3x320x320xf32>, tensor<1x1xi32>, tensor<1x3x32x32xf32>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
  // CHECK-LABEL: func.func @scatter_simple_1
  // CHECK: "ttnn.reshape"({{.*}}) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}>
  // CHECK-SAME: (tensor<1x1xsi32, {{.*}}>) -> tensor<1x1x1x1xsi32, {{.*}}>
  // CHECK: "ttnn.repeat"({{.*}}) <{repeat_dims = #ttnn.shape<1x3x32x32>}>
  // CHECK: "ttnn.scatter"({{.*}}) <{cq_id = 0 : ui32, dim = 0 : i32}>
  // CHECK-SAME: (tensor<1x3x320x320xf32, {{.*}}>, tensor<1x3x32x32xsi32, {{.*}}>, tensor<1x3x32x32xf32, {{.*}}>) -> tensor<1x3x320x320xf32, {{.*}}>
  return %1 : tensor<1x3x320x320xf32>
}

func.func @scatter_simple_2(%arg0: tensor<32x32xi32>, %arg1: tensor<16x1xi32>, %arg2: tensor<16x32xi32>) -> tensor<32x32xi32> {
  %0 = ttir.empty() : tensor<32x32xi32>
  %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 1>}> : (tensor<32x32xi32>, tensor<16x1xi32>, tensor<16x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
  // CHECK-LABEL: func.func @scatter_simple_2
  // CHECK: "ttnn.repeat"({{.*}}) <{repeat_dims = #ttnn.shape<1x32>}>
  // CHECK: "ttnn.scatter"({{.*}}) <{cq_id = 0 : ui32, dim = 0 : i32}>
  // CHECK-SAME: (tensor<32x32xsi32, {{.*}}>, tensor<16x32xsi32, {{.*}}>, tensor<16x32xsi32, {{.*}}>) -> tensor<32x32xsi32, {{.*}}>
  return %1 : tensor<32x32xi32>
}
