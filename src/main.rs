use nnetwork::*;
use tch::nn::OptimizerConfig;
use anyhow::{Ok, Result};
//construct model X
//train model X
//save model X
//load model X
//model accuracy X
//test model X
//CLI
pub fn main() -> Result<()> {
    // let _r = test();

    let mut vs = tch::nn::VarStore::new(tch::Device::cuda_if_available());
    // let model = construct_model(&vs.root());
    // let data = tch::vision::cifar::load_dir("data")?;
    let model = fast_resnet(&vs.root());
    load_model(&mut vs, "models/fastnet1.model")?;
    // let mut optimizer = tch::nn::Adam::default().build(&vs, 0.)?;
    // train_model(&vs.root(), &model, &mut optimizer, &data,2);
    // println!("{:#?}",model);
    // println!("{:?}",accuracy_model(&model, &data, &vs.device()));
    let _res = predict(&model, "test/golden-retriever-dog.jpg",&vs.device())?;
    let _res = predict(&model, "test/German_Shepherd.jpg",&vs.device())?;
    let _res = predict(&model, "test/big_jazz.jpg",&vs.device())?;
    let _res = predict(&model, "test/small_jazz.png",&vs.device())?;
    let _res = predict(&model, "test/truck.jfif",&vs.device())?;
    Ok(())
}
