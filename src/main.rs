use nnetwork::*;
use tch::nn::OptimizerConfig;
use tch::nn::ModuleT;
//construct model X
//train model X
//save model 
//load model
//model accuracy
pub fn main() -> () {
    // let _r = test();

    let vs = tch::nn::VarStore::new(tch::Device::cuda_if_available());

    // let model = construct_model(&vs.root());
    let data = tch::vision::cifar::load_dir("data").unwrap();
    let model = fast_resnet(&vs.root());
    let mut optimizer = tch::nn::Adam::default().build(&vs, 0.).unwrap();
    train_model(&vs.root(), &model, &mut optimizer, &data);
}
