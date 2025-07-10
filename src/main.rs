use std::time::Instant;

use tch::{self, Kind, Tensor};

use ort::{
    execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
};

fn tch_demo() {
    tch::set_num_interop_threads(1); // for throughput
    tch::set_num_threads(1); //for throughput
    let pt_path = "/root/projects/tch-deploy/traced_model.pt";
    let device = tch::Device::Cuda(0);
    let mut model = tch::CModule::load_on_device(pt_path, device).unwrap();
    model.set_eval();
    let mut filnal_result = 0.0;
    {
        let _guard = tch::no_grad_guard();

        for _ in 0..100 {
            let feat = ndarray::Array3::from_elem((256, 10, 10), 0.2_f32);
            let (bs, ts, f_len) = feat.dim();
            let feat_vec: Vec<f32> = feat.into_raw_vec_and_offset().0;
            let inp = tch::Tensor::from_slice(&feat_vec)
                .view((bs as i64, ts as i64, f_len as i64))
                .to_device(device);

            let res = model.forward_ts(&[inp]).unwrap();
            let mut res_vec: Vec<f32> = vec![0.0_f32; res.numel()];
            res.copy_data(&mut res_vec, res.numel());
            filnal_result += res_vec[0];
        }
        println!("tch {}", filnal_result);
        filnal_result = 0.0;

        let instance = Instant::now();

        for _ in 0..10000 {
            let feat = ndarray::Array3::from_elem((256, 10, 10), 0.2_f32);
            let (bs, ts, f_len) = feat.dim();
            let feat_vec: Vec<f32> = feat.into_raw_vec_and_offset().0;
            let inp = tch::Tensor::from_slice(&feat_vec)
                .view((bs as i64, ts as i64, f_len as i64))
                .to_device(device);

            let res = model.forward_ts(&[inp]).unwrap();
            let mut res_vec: Vec<f32> = vec![0.0_f32; res.numel()];
            res.copy_data(&mut res_vec, res.numel());
            filnal_result += res_vec[0];
        }
        println!(
            "tch {}, secs:{}",
            filnal_result,
            instance.elapsed().as_secs_f32()
        );
    }
}

fn ort_demo() {
    // ort::init()
    //     .with_name("demo")
    //     .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])
    //     .commit()
    //     .unwrap();

    ort::init()
        .with_name("demo")
        .with_execution_providers([TensorRTExecutionProvider::default()
            .with_device_id(1)
            .build()
            .error_on_failure()])
        .commit()
        .unwrap();

    let session = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_intra_threads(1)
        .unwrap()
        .with_inter_threads(1)
        .unwrap()
        .commit_from_file("/root/projects/tch-deploy/model.onnx")
        .unwrap();
    let mut result = 0.0;

    for _ in 0..100 {
        let feat = ndarray::Array3::from_elem((256, 10, 10), 0.2_f32);
        // let lengths = ndarray::Array1::from_vec(vec![9_i64, 8, 8]);
        let output = session.run(ort::inputs!["feat"=>feat].unwrap()).unwrap();
        let output = output
            .get("prob")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap();
        result += output[[0, 0]];
    }
    println!("ort, warm up: {}", result);

    let mut result = 0.0;
    let instance = Instant::now();
    for _ in 0..10000 {
        let feat = ndarray::Array3::from_elem((256, 10, 10), 0.2_f32);
        // let lengths = ndarray::Array1::from_vec(vec![9_i64, 8, 8]);
        let output = session.run(ort::inputs!["feat"=>feat].unwrap()).unwrap();
        let output = output
            .get("prob")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap();
        result += output[[0, 0]];
    }
    println!("ort: {}, secs:{}", result, instance.elapsed().as_secs_f32());
}
fn main() {
    ort_demo();
    tch_demo();
}
