use tch::{self, Device, Kind, Tensor};

fn main() {
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
