use flate2;
use std::fs;
use std::time::Instant;
use tch;

pub mod rayon_demos;

use ort::{
    execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

fn tch_init() {
    tch::set_num_interop_threads(1); // for throughput
    tch::set_num_threads(1); //for throughput
}

fn tch_fp32() {
    let nn_model_path = "/root/models/2025Q3-cnn-dw-residual-nobias-stage1-v3-data-epo17.tar.gz";
    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp_dir_path = tmp_dir.path().to_path_buf();
    let tar_gz_file =
        fs::File::open(nn_model_path).expect(&format!("model not found: {}", nn_model_path));
    let tar_file = flate2::read::GzDecoder::new(tar_gz_file);
    let mut archive = tar::Archive::new(tar_file);
    archive.unpack(tmp_dir_path.to_str().unwrap()).unwrap();

    let pt_path = tmp_dir_path.join("model");

    let device = tch::Device::Cuda(3);
    let mut model = tch::CModule::load_on_device(pt_path, device).unwrap();
    model.set_eval();
    let mut result = 0.0;
    {
        let _guard = tch::no_grad_guard();
        let instance = Instant::now();

        for _ in 0..20 {
            let feat = ndarray::Array3::from_elem((128, 200, 61), 0.0_f32);
            let (bs, ts, f_len) = feat.dim();
            let feat_vec: Vec<f32> = feat.into_raw_vec_and_offset().0;
            let lengths = ndarray::Array1::from_elem(128, 200_i64);

            let inp = tch::Tensor::from_slice(&feat_vec)
                .view((bs as i64, ts as i64, f_len as i64))
                .to_device(device);
            let lengths =
                tch::Tensor::from_slice(&lengths.into_raw_vec_and_offset().0).to_device(device);

            let res = model.forward_ts(&[inp, lengths]).unwrap();
            let mut res_vec: Vec<f32> = vec![0.0_f32; res.numel()];
            res.copy_data(&mut res_vec, res.numel());
            result += res_vec[0];
        }
        println!(
            "tch: warmup {}, secs:{}",
            result,
            instance.elapsed().as_secs_f32()
        );
        result = 0.0;

        let instance = Instant::now();

        for _ in 0..1000 {
            let feat = ndarray::Array3::from_elem((128, 200, 61), 0.0_f32);
            let lengths = ndarray::Array1::from_elem(128, 200_i64);
            let (bs, ts, f_len) = feat.dim();
            let feat_vec: Vec<f32> = feat.into_raw_vec_and_offset().0;

            let inp = tch::Tensor::from_slice(&feat_vec)
                .view((bs as i64, ts as i64, f_len as i64))
                .to_device(device);

            let lengths =
                tch::Tensor::from_slice(&lengths.into_raw_vec_and_offset().0).to_device(device);

            let res = model.forward_ts(&[inp, lengths]).unwrap();
            let mut res_vec: Vec<f32> = vec![0.0_f32; res.numel()];
            res.copy_data(&mut res_vec, res.numel());
            result += res_vec[0];
        }
        println!("tch {}, secs:{}", result, instance.elapsed().as_secs_f32());
    }
}

fn tch_half() {
    let nn_model_path = "/root/models/2025Q3-cnn-dw-residual-nobias-stage1-v3-data-epo17-fp16.tar.gz";
    let tmp_dir = tempfile::tempdir().unwrap();
    let tmp_dir_path = tmp_dir.path().to_path_buf();
    let tar_gz_file =
        fs::File::open(nn_model_path).expect(&format!("model not found: {}", nn_model_path));
    let tar_file = flate2::read::GzDecoder::new(tar_gz_file);
    let mut archive = tar::Archive::new(tar_file);
    archive.unpack(tmp_dir_path.to_str().unwrap()).unwrap();

    let pt_path = tmp_dir_path.join("model");
    let device = tch::Device::Cuda(3);

    let mut model = tch::CModule::load_on_device(pt_path, device).unwrap();
    model.set_eval();
    let mut result = 0.0;
    {
        let _guard = tch::no_grad_guard();
        let instance = Instant::now();

        for _ in 0..20 {
            let feat = ndarray::Array3::from_elem((128, 200, 61), 0.0_f32);
            let (bs, ts, f_len) = feat.dim();
            let feat_vec: Vec<f32> = feat.into_raw_vec_and_offset().0;
            let lengths = ndarray::Array1::from_elem(128, 200_i64);

            let inp = tch::Tensor::from_slice(&feat_vec)
                .view((bs as i64, ts as i64, f_len as i64))
                .to_device(device);
            let lengths =
                tch::Tensor::from_slice(&lengths.into_raw_vec_and_offset().0).to_device(device);

            let res = model
                .forward_ts(&[inp.to_dtype(tch::Kind::BFloat16, true, true), lengths])
                .unwrap()
                .to_dtype(tch::Kind::Float, true, true);
            let mut res_vec: Vec<f32> = vec![0.0_f32; res.numel()];
            res.copy_data(&mut res_vec, res.numel());
            result += res_vec[0];
        }
        println!(
            "tch_half warmup {}, secs:{}",
            result,
            instance.elapsed().as_secs_f32()
        );
        result = 0.0;

        let instance = Instant::now();
        let num_iter = 1000;

        for _ in 0..num_iter {
            let feat = ndarray::Array3::from_elem((128, 200, 61), 0.0_f32);
            let lengths = ndarray::Array1::from_elem(128, 200_i64);
            let (bs, ts, f_len) = feat.dim();
            let feat_vec: Vec<f32> = feat.into_raw_vec_and_offset().0;

            let inp = tch::Tensor::from_slice(&feat_vec)
                .view((bs as i64, ts as i64, f_len as i64))
                .to_device(device);

            let lengths =
                tch::Tensor::from_slice(&lengths.into_raw_vec_and_offset().0).to_device(device);

            let res = model
                .forward_ts(&[inp.to_dtype(tch::Kind::BFloat16, true, true), lengths])
                .unwrap()
                .to_dtype(tch::Kind::Float, true, true);

            let mut res_vec: Vec<f32> = vec![0.0_f32; res.numel()];
            res.copy_data(&mut res_vec, res.numel());
            result += res_vec[0];
        }
        println!(
            "tch_half {}, secs:{}",
            result,
            instance.elapsed().as_secs_f32()
        );
    }
}

fn ort_demo() {
    ort::init()
        .with_name("demo")
        .with_execution_providers([CUDAExecutionProvider::default()
            .with_device_id(1)
            .build()
            .error_on_failure()])
        .commit()
        .unwrap();

    // ort::init()
    //     .with_name("demo")
    //     .with_execution_providers([TensorRTExecutionProvider::default()
    //         .with_device_id(1)
    //         .with_fp16(true)
    //         .build()
    //         .error_on_failure()])
    //     .commit()
    //     .unwrap();

    let mut session = Session::builder()
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
    let instance = Instant::now();

    for _ in 0..20 {
        let feat = ndarray::Array3::from_elem((128, 200, 61), 0.0_f32);
        let lengths = ndarray::Array1::from_elem(128, 200_i64);
        let output = session
            .run(ort::inputs!["feature"=>TensorRef::from_array_view(&feat).unwrap(), "length"=> TensorRef::from_array_view(&lengths).unwrap()])
            .unwrap();
        let output = output
            .get("probs")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap();
        result += output.1[0];
    }
    println!(
        "ort: warm up, {}, secs:{}",
        result,
        instance.elapsed().as_secs_f32()
    );

    let mut result = 0.0;
    let instance = Instant::now();
    for _ in 0..1000 {
        let feat = ndarray::Array3::from_elem((128, 200, 61), 0.0_f32);
        let lengths = ndarray::Array1::from_elem(128, 200_i64);
        let output = session
            .run(ort::inputs!["feature"=>TensorRef::from_array_view(&feat).unwrap(), "length"=> TensorRef::from_array_view(&lengths).unwrap()])
            .unwrap();

        let output = output
            .get("probs")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap();
        result += output.1[0];
    }
    println!("ort: {}, secs:{}", result, instance.elapsed().as_secs_f32());
}

fn main() {
    tch_init();
    tch_fp32();
    tch_half();
    // ort_demo();
    // tch_demo();
}
