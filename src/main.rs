use tch::{self, Device, Kind, Tensor};

use ndarray::{Axis, Ix2};
use ort::{
    execution_providers::CUDAExecutionProvider,
    session::{builder::GraphOptimizationLevel, output, Session},
    value::TensorRef,
    Error,
};

fn tch_demo() {
    let pt_path = "/root/projects/tch-deploy/traced_model.pt";
    let mut model = tch::CModule::load(pt_path).unwrap();
    model.set_eval();

    let inp = Tensor::rand([3, 10, 10], (Kind::Float, Device::Cpu));
    let lengths = Tensor::from_slice(&[8, 9, 9]);

    let res = {
        let _guard = tch::no_grad_guard();
        let res = model.forward_ts(&[inp, lengths]).unwrap();
        res
    };

    res.print();
    let mut res_vec: Vec<f32> = vec![0.0_f32; res.numel()];
    res.copy_data(&mut res_vec, res.numel());
    println!("{:?}", res_vec);
}

fn ort_demo() {
    ort::init()
        .with_name("demo")
        .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])
        .commit()
        .unwrap();

    let session = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level1)
        .unwrap()
        .with_intra_threads(1)
        .unwrap()
        .commit_from_file("/root/projects/tch-deploy/model.onnx")
        .unwrap();
    let feat = ndarray::Array3::from_elem((2, 10, 10), 0.2_f32);
    let lengths = ndarray::Array1::from_vec(vec![9_i64, 8]);
    let output = session
        .run(ort::inputs!["feat"=>feat, "len"=>lengths].unwrap())
        .unwrap();
    let output = output.get("prob").unwrap().try_extract_tensor::<f32>().unwrap();
    println!("{:?}", output);
}
fn main() {
    ort_demo();
}
