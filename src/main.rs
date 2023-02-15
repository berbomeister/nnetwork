use std::{fs::File, io::{IoSlice, Write, Read, BufRead}};

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

    // cli()?;

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
    
    let mut vs = tch::nn::VarStore::new(tch::Device::cuda_if_available());

    let (model,stack) = construct_model(&vs.root());
    let modelname = "architectures/test1";
    
    //save model
    let mut file = File::create(modelname).expect("Unable to create file for model architecture.");
    println!("{:#?}",stack);
    let ser = serde_json::to_string(&stack)?;
    println!("{}",ser.as_str());
    file.write(ser.as_bytes())?;

    //load model
    let mut buf =String::new();
    File::open(modelname)?.read_to_string(&mut buf)?;
    println!("{}",buf);
    let deser :Vec<Layer> = serde_json::from_str(buf.as_str())?;
    println!("{:#?}",deser);

    
    // for layer in stack{
    //     let serialized = serde_json::to_string(&layer)?;
    //     file.write(serialized.as_bytes())?;
    //     file.write(b"\n")?;
    // }
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
// add conv_layer 3 64 3 --default
// add conv_layer 64 256 3 --default
// add maxpool 4
// add dropout 0.3
// add conv_layer 256 512 3 --default
// add maxpool 4
// add dropout 0.2
// add flatten
// add linear -1 10
// build