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

    cli()?;

    // let str = "models/fastnet1.model";
    // let t = str.to_string().split(".").collect::<Vec<&str>>()[0].to_string();
    // let name = t.split('/').collect::<Vec<&str>>()[1];
    // println!("{:?}",name);
    // println!("{:?}",modelname(str));
    // println!("{}","123".parse::<i64>()?);
    // let welcome = std::fs::read_to_string("txt/welcome")?;
    // println!("{welcome}");
    // let args: Vec<String> = std::env::args().collect();
    // println!("{:#?}",args);
    // if args.len() == 1 {
    //     // println!("asdasdasd");
    //     cli()?;
    // }
    // else {
    //     //read arguments and do action
    // }
    
    // let mut vs = tch::nn::VarStore::new(tch::Device::cuda_if_available());

    // let model = construct_model(&vs.root());
    // let model = fast_resnet(&vs.root());
    // load_model(&mut vs, "models/fastnet1.model")?;

    // let data = tch::vision::cifar::load_dir("data")?;
    // let mut optimizer = tch::nn::Adam::default().build(&vs, 0.)?;
    // train_model(&vs.root(), &model, &mut optimizer, &data,2);
    // println!("{:#?}",model);
    // println!("{:?}",accuracy_model(&model, &data, &vs.device()));

    // let _res = predict(&model, "test/golden-retriever-dog.jpg",&vs.device())?;
    // let _res = predict(&model, "test/German_Shepherd.jpg",&vs.device())?;
    // let _res = predict(&model, "test/big_jazz.jpg",&vs.device())?;
    // let _res = predict(&model, "test/small_jazz.png",&vs.device())?;
    // let _res = predict(&model, "test/truck.jfif",&vs.device())?;
    Ok(())
}
