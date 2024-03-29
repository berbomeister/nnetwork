use anyhow::{Ok, Result};
use serde::{Deserialize, Serialize};
use std::{
    fmt::Display,
    fs,
    fs::File,
    io::{Read, Write},
};
use tch::nn::{ModuleT, Optimizer, OptimizerConfig, SequentialT};
use tch::vision::dataset::Dataset;
use tch::{nn, Device};

pub fn learning_rate(epoch: i64) -> f64 {
    if epoch < 15 {
        0.1
    } else if epoch < 30 {
        0.01
    } else {
        0.005
    }
}
pub fn conv2d_sublayer(
    vs: &nn::Path,
    in_channels: i64,
    out_channels: i64,
    kernel_size: i64,
    stride: Option<i64>,
    padding: Option<i64>,
    bias: bool,
) -> SequentialT {
    let padding = if let Some(val) = padding {
        val
    } else {
        0 //default value
    };
    let stride = if let Some(val) = stride {
        val
    } else {
        1 //default value
    };

    let conv2d_cfg = nn::ConvConfig {
        stride: stride,
        padding: padding,
        bias: bias,
        ..Default::default()
    };
    nn::seq_t()
        .add(nn::conv2d(
            vs,
            in_channels,
            out_channels,
            kernel_size,
            conv2d_cfg,
        ))
        .add_fn(|x| x.relu())
        .add(nn::batch_norm2d(vs, out_channels, Default::default()))
}

pub fn layer<'a>(
    vs: &nn::Path,
    in_channels: i64,
    out_channels: i64,
    kernel_size: i64,
    stride: Option<i64>,
    padding: Option<i64>,
    bias: bool,
    maxpool_kernel: i64,
    dropout: Option<f64>,
) -> nn::FuncT<'a> {
    let pre = conv2d_sublayer(
        &vs.sub("pre"),
        in_channels,
        out_channels / 2,
        kernel_size,
        stride,
        padding,
        bias,
    );
    let post = conv2d_sublayer(
        &vs.sub("pre"),
        out_channels / 2,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
    );
    nn::func_t(move |x, train| {
        let res_ = x.apply_t(&pre, train);
        let res = res_
            .apply_t(&post, train)
            .max_pool2d_default(maxpool_kernel);
        match dropout {
            Some(p) => res.dropout(p, train),
            _ => res,
        }
    })
}

pub fn cnn1(vs: &nn::Path) -> SequentialT {
    nn::seq_t()
        .add(layer(
            &vs.sub("layer1"),
            3,
            64,
            3,
            Some(1),
            Some(1),
            true,
            2,
            Some(0.25),
        ))
        .add(layer(
            &vs.sub("layer2"),
            64,
            128,
            3,
            Some(1),
            Some(1),
            true,
            2,
            Some(0.25),
        ))
        .add(layer(
            &vs.sub("layer3"),
            128,
            256,
            3,
            Some(1),
            Some(1),
            true,
            2,
            Some(0.25),
        ))
        .add_fn(|x| x.flatten(1, 3))
        .add(nn::linear(
            &vs.sub("output1"),
            256 * 4 * 4,
            512,
            Default::default(),
        ))
        // .add_fn(|x| x.relu())
        .add(nn::linear(&vs.sub("output2"), 512, 10, Default::default()))
}
pub fn fast_resnet2(vs: &nn::Path) -> SequentialT {
    nn::seq_t()
        .add(layer(
            &vs.sub("layer1"),
            3,
            256,
            3,
            None,
            Some(1),
            true,
            4,
            Some(0.25),
        ))
        .add(layer(
            &vs.sub("layer2"),
            256,
            512,
            3,
            None,
            Some(1),
            true,
            2,
            Some(0.25),
        ))
        .add_fn(|x| x.max_pool2d_default(4).flat_view())
        .add(nn::linear(vs.sub("linear"), 512, 10, Default::default()))
        .add_fn(|x| x * 0.125)
}
pub fn fast_resnet(vs: &nn::Path) -> SequentialT {
    nn::seq_t()
        .add(conv2d_sublayer(
            &vs.sub("pre"),
            3,
            64,
            3,
            None,
            Some(1),
            true,
        ))
        .add(conv2d_sublayer(
            &vs.sub("inter"),
            64,
            256,
            3,
            None,
            Some(1),
            true,
        ))
        .add_fn(|x| x.max_pool2d_default(4))
        .add_fn_t(|x, train| x.dropout(0.25, train))
        .add(conv2d_sublayer(
            &vs.sub("inter2"),
            256,
            512,
            3,
            None,
            Some(1),
            true,
        ))
        .add_fn(|x| x.max_pool2d_default(2))
        .add_fn(|x| x.max_pool2d_default(4).flat_view())
        .add(nn::linear(vs.sub("linear"), 512, 10, Default::default()))
        .add_fn(|x| x * 0.125)
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Layer {
    ConvLayer(i64, i64, i64, Option<i64>, Option<i64>, bool),
    Maxpool(i64),
    Dropout(f64),
    Flatten(),
    Linear(i64, i64),
    Relu(),
    Other(String),
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Layers(pub Vec<Layer>);
#[derive(Debug)]
enum Output {
    Conv(Option<i64>, i64, i64),
    Linear(i64),
}
#[derive(Debug)]
struct Model {
    stack: Layers,
    output: Output,
}
impl Model {
    fn new() -> Model {
        Model {
            stack: Layers(Vec::new()),
            output: Output::Conv(None, 32, 32),
        }
    }
}
/// ## Constructs a model from command-line commands. Valid commands are:   
/// - ### add conv_layer in channels out channels kernel_size \[--default |  stride padding\]
///
///     -> adds a convolution layer to the architecture with the specified params; default value for (stride, padding) is (1,0)
///
/// - ### add dropout p
///
///     -> adds a dropout layer with p probabilty
///
/// - ### add maxpool kernel_size
///
///     -> adds a maxpool layer to the architecture with kernel size and no stride/padding
///
/// - ### add flatten
///
///     -> adds a flatten layer to the architecture
///
/// - ### add linear in_features out_features
///
///     -> adds a linear layer to the architecture with input_features and output_features
///
/// - ### build
///     -> builds and returns the model
///
pub fn construct_model(vs: &nn::Path) -> Option<(SequentialT, Layers)> {
    let mut model = Model::new();
    loop {
        let mut user_input = String::new();
        let stdin = std::io::stdin();
        let _e = stdin.read_line(&mut user_input);
        let input = &user_input.trim().split(" ").collect::<Vec<&str>>();
        if input[0] == "build" {
            break;
        } else if input[0] == "help" {
            println!("Available commands are \"add\" and \"build\" and \"abort\".
            possible arguments for add are:
            add conv_layer in_c out_c k_size [\"--default\" | stride padding] :
                Adds a convolution layer with in_c in channels, out_c output channels, k_size kernel size and stride and padding (default values are 0, 1)
            add maxpool k_size :
                Adds a 2D maxpool layer with kernel size (k_size,k,size)
            add dropout p :
                Adds a Dropout layer with dropout probability = p
            add linear in_f out_f :
                Adds a Linear layer with in_f in features and out_f output features
            add flatten :
                Adds a layer that flattens all dimensions, except the first, which is understood as batch size
            add relu :
                Adds an activation layer for the function relu(x) = max(0,x)");
        } else if input[0] == "add" {
            assert!(
                input.len() >= 2,
                "{:?} - missing arguments",
                input.join(" ")
            );
            if input[1] == "conv_layer" {
                if input.len() < 6 || input.len() > 7 {
                    println!("Incorrect number of arguments!\n Correct use is: \"add conv_layer in_channels out_channels kernel_size [--default | stride padding]\"");
                    continue;
                }
                let in_channels = i64::from_str_radix(input[2], 10).unwrap();
                let out_channels = i64::from_str_radix(input[3], 10).unwrap();
                let kernel_size = i64::from_str_radix(input[4], 10).unwrap();
                let (stride, padding) = if input[5] == "--default" {
                    (None, Some(1))
                } else {
                    (
                        Some(i64::from_str_radix(input[5], 10).unwrap()),
                        Some(i64::from_str_radix(input[6], 10).unwrap()),
                    )
                };
                match model.output {
                    Output::Conv(None, h, w) => {
                        //first conv layer
                        if in_channels != 3 {
                            println!("Input tensor has size (3,32,32), first layer must have 3 input channels!");
                            continue;
                        }
                        model.stack.0.push(Layer::ConvLayer(
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride,
                            padding,
                            true,
                        ));
                        model.output = Output::Conv(Some(out_channels), h, w);
                    }
                    Output::Conv(Some(out_dim), h, w) => {
                        if out_dim == in_channels {
                            // new layer input dim matches last layer output dim
                            model.stack.0.push(Layer::ConvLayer(
                                in_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                padding,
                                true,
                            ));
                            model.output = Output::Conv(Some(out_channels), h, w);
                        } else {
                            println!("Input dim does not match last layer's output dim!\n input : {}, output : {}",in_channels,out_dim);
                            continue;
                        }
                    }
                    Output::Linear(_out_features) => {
                        println!("Cannot put a conv layer after a linear layer!");
                    }
                };
            } else if input[1] == "maxpool" {
                if input.len() != 3 {
                    println!("Incorrect number of arguments!\n Correct use is: \"add maxpool kernel_size\"");
                    continue;
                }
                let kernel_size = i64::from_str_radix(input[2], 10).unwrap();
                match model.output {
                    Output::Conv(None, h, w) => {
                        //first layer
                        if h % kernel_size != 0 || w % kernel_size != 0 {
                            println!("Maxpool layer's kernel size is incompatible with the model's dimensions!");
                            continue;
                        }
                        model.stack.0.push(Layer::Maxpool(kernel_size));
                        model.output = Output::Conv(None, h / kernel_size, w / kernel_size);
                    }
                    Output::Conv(Some(out_dim), h, w) => {
                        //after conv_layer
                        if h % kernel_size != 0 || w % kernel_size != 0 {
                            println!("Maxpool layer's kernel size is incompatible with the model's dimensions!");
                            continue;
                        }
                        model.stack.0.push(Layer::Maxpool(kernel_size));
                        model.output =
                            Output::Conv(Some(out_dim), h / kernel_size, w / kernel_size);
                    }
                    Output::Linear(_out_features) => {
                        println!("Cannot add maxpool layer after a linear layer!");
                        continue;
                    }
                };
            } else if input[1] == "dropout" {
                if input.len() != 3 {
                    println!("Incorrect number of arguments!\n Correct use is: \"add dropout p\"");
                    continue;
                }
                let dropout = (input[2]).parse::<f64>().unwrap();
                if dropout <= 0. || dropout >= 1. {
                    println!("Dropout probability should be between 0 and 1.");
                    continue;
                }
                model.stack.0.push(Layer::Dropout(dropout));
            } else if input[1] == "linear" {
                if input.len() != 4 {
                    println!("Incorrect number of arguments!\n Correct use is: \"add linear in_features out_features\"");
                    continue;
                }
                let in_features = (input[2]).parse::<i64>().unwrap();
                let out_features = (input[3]).parse::<i64>().unwrap();
                match model.output {
                    Output::Conv(_, _, _) => {
                        //first layer
                        println!("Cannot put Linear layer as first layer, or directly after a conv layer!\n Consider adding a flatten layer first.");
                    }
                    Output::Linear(out_features_last) => {
                        if in_features == out_features_last || in_features == -1 {
                            model
                                .stack
                                .0
                                .push(Layer::Linear(out_features_last, out_features));
                            model.output = Output::Linear(out_features);
                        } else {
                            println!("Input dim does not match last layer's output dim!\n input : {}, output : {}",in_features,out_features_last);
                            continue;
                        };
                    }
                };
            } else if input[1] == "flatten" {
                if input.len() != 2 {
                    println!("Flatten does not take arguments!\n Correct use is: \"add flatten\"");
                    continue;
                }
                match model.output {
                    Output::Conv(None, h, w) => {
                        //first layer
                        model.stack.0.push(Layer::Flatten());
                        model.output = Output::Linear(3 * h * w);
                    }
                    Output::Conv(Some(out_dim), h, w) => {
                        model.stack.0.push(Layer::Flatten());
                        model.output = Output::Linear(out_dim * h * w);
                    }
                    Output::Linear(_out_features) => {
                        println!("No point in adding a flatten after a linear layer. This command is skipped.");
                        continue;
                    }
                };
            } else if input[1] == "relu" {
                if input.len() != 2 {
                    println!("ReLu does not take arguments!\n Correct use is: \"add relu\"");
                    continue;
                }
                model.stack.0.push(Layer::Relu());
            } else {
                println!("Unknown layer name!");
                continue;
            }
        } else if input[0] == "abort"{
            return None;
        } else {
            println!("Unknown command \"{}\"! Valid commands are:\n add layer_name [layer_options]\n build",input[0])
        }
    }
    let net = model
        .stack
        .0
        .iter()
        .fold(tch::nn::seq_t(), move |model, layer| match *layer {
            Layer::ConvLayer(in_channels, out_channels, kernel_size, stride, padding, bias) => {
                model.add(conv2d_sublayer(
                    vs,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias,
                ))
            }
            Layer::Maxpool(kernel) => model.add_fn(move |x| x.max_pool2d_default(kernel)),
            Layer::Dropout(dropout) => model.add_fn_t(move |x, train| x.dropout(dropout, train)),
            Layer::Flatten() => model.add_fn(|x| x.flat_view()),
            Layer::Linear(in_channels, out_channels) => model.add(tch::nn::linear(
                vs,
                in_channels,
                out_channels,
                Default::default(),
            )),
            Layer::Relu() => model.add_fn(|x| x.relu()),
            _ => model,
        });
    Some((net, model.stack))
}

pub fn train_model(
    vs: &nn::Path,
    model: &dyn ModuleT,
    optimizer: &mut Optimizer,
    data: &Dataset,
    epochs: i64,
) {
    for epoch in 0..epochs {
        optimizer.set_lr(learning_rate(epoch));
        for (_i, (bimages, blabels)) in data
            .train_iter(64)
            .shuffle()
            .to_device(vs.device())
            .enumerate()
        {
            let bimages = tch::vision::dataset::augmentation(&bimages, true, 4, 8);
            let pred = model.forward_t(&bimages, true);
            let loss = pred.cross_entropy_for_logits(&blabels);
            optimizer.backward_step(&loss);
        }
        let test_accuracy =
            model.batch_accuracy_for_logits(&data.test_images, &data.test_labels, vs.device(), 512);
        println!(
            "epoch: {:4} test acc: {:5.2}%",
            epoch + 1,
            100. * test_accuracy,
        );
    }
}

pub fn save_model(vs: &nn::VarStore, filename: &str) -> Result<()> {
    vs.save(filename)?;
    Ok(())
}
pub fn load_model(vs: &mut nn::VarStore, filename: &str) -> Result<()> {
    vs.load(filename)?;
    Ok(())
}
pub fn load_net(modelname: &str) -> Result<Layers> {
    let mut buf = String::new();
    let path = format!("architectures/{modelname}");
    File::open(path)?.read_to_string(&mut buf)?;
    let deser: Layers = serde_json::from_str(buf.as_str())?;
    Ok(deser)
}
pub fn save_net(stack: &Layers, modelname: &str) -> Result<()> {
    let path = format!("architectures/{modelname}");
    let mut file = File::create(path)?;
    let ser = serde_json::to_string(&stack)?;
    file.write(ser.as_bytes())?;
    Ok(())
}
pub fn accuracy_model(model: &dyn ModuleT, data: &Dataset, device: &Device) -> f64 {
    let _no_grad = tch::no_grad_guard();
    let mut sum_accuracy = 0f64;
    let mut sample_count = 0f64;
    let batch_size = 64;
    for (xs, ys) in data
        .test_iter(batch_size)
        .to_device(*device)
        .return_smaller_last_batch()
    {
        let acc = model.forward_t(&xs, false).accuracy_for_logits(&ys);
        let size = xs.size()[0] as f64;
        sum_accuracy += f64::from(&acc) * size;
        sample_count += size;
    }
    sum_accuracy / sample_count
}
pub fn predict(model: &dyn ModuleT, imagepath: &str, device: &Device) -> Result<String> {
    let _no_grad = tch::no_grad_guard();
    let image = tch::vision::image::load(imagepath)?;
    let classes = vec![
        "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
    ];
    let image = tch::vision::image::resize(&image, 32, 32)?
        .unsqueeze(0)
        .to_kind(tch::Kind::Float)
        .to_device(*device)
        / 255;

    let (_value, index) = model
        .forward_t(&image, false)
        .softmax(-1, tch::Kind::Float)
        .topk(1, -1, true, true);
    Ok(String::from(classes[index.int64_value(&[0]) as usize]))
}

pub fn cli() -> Result<()> {
    let welcome = fs::read_to_string("txt/welcome")?;
    println!("{welcome}");
    let valid_commands = vec![
        "help",
        "load",
        "save",
        "construct",
        "train",
        "accuracy",
        "predict",
        "view",
    ];
    let data = tch::vision::cifar::load_dir("data")?;
    let mut vs = tch::nn::VarStore::new(tch::Device::cuda_if_available());
    let mut loaded_model = false;
    let mut net = nn::seq_t();
    let mut stack: Layers = Layers(Vec::new());
    loop {
        let mut user_input = String::new();
        print!(">>");
        std::io::stdout().flush()?;
        let stdin = std::io::stdin();
        let _e = stdin.read_line(&mut user_input);
        let input = &user_input.trim().split(" ").collect::<Vec<&str>>();
        match input[0] {
            //help [command]
            "help" => {
                if input.len() == 1 {
                    let help = fs::read_to_string("txt/help/all")?;
                    println!("{help}");
                } else {
                    if valid_commands.contains(&input[1]) {
                        let help = fs::read_to_string(format!("txt/help/{}", input[1]))?;
                        println!("{help}");
                    } else {
                        println!("Invalid command name. Type help for more information");
                        continue;
                    }
                };
            }
            //load modelname
            "load" => {
                if input.len() != 2 {
                    println!("Wrong syntax. Type help load for more information");
                    continue;
                }
                let modelname = input[1];
                let path = format!("models/{modelname}.model");
                let pre_trained = vec!["cnn1", "fastnet1", "fastnet2"];
                vs = tch::nn::VarStore::new(tch::Device::cuda_if_available());
                if pre_trained.contains(&modelname) {
                    match modelname {
                        "cnn1" => {
                            net = cnn1(&vs.root());
                            stack = Layers(vec![Layer::Other(String::from("cnn1"))]);
                        }
                        "fastnet1" => {
                            net = fast_resnet(&vs.root());
                            stack = Layers(vec![Layer::Other(String::from("fastnet1"))]);
                        }
                        "fastnet2" => {
                            net = fast_resnet2(&vs.root());
                            stack = Layers(vec![Layer::Other(String::from("fastnet2"))]);
                        }
                        _ => (),
                    }
                } else {
                    stack = load_net(&modelname)?;
                    net = from_stack(&stack, &vs.root());
                }
                load_model(&mut vs, &path)?;
                loaded_model = true;
                println!("Model successfully loaded.");
            }
            // save modelname
            "save" => {
                if input.len() != 2 {
                    println!("Wrong syntax. Type help save for more information.");
                    continue;
                }
                if !loaded_model {
                    println!("No loaded model, ignoring this command!");
                    continue;
                }
                let modelname = input[1];
                let path = format!("models/{modelname}.model");
                let pre_trained = vec!["cnn1", "fastnet1", "fastnet2"];
                if pre_trained.contains(&modelname) {
                    save_model(&vs, &path)?;
                    save_net(&stack, modelname)?;
                } else {
                    save_model(&vs, &path)?;
                    save_net(&stack, modelname)?;
                }
                println!("Model successfully saved.");
            }
            //construct
            "construct" => {
                if input.len() != 1 {
                    println!("This command does not take any arguments, they will be ignored!");
                }
                vs = tch::nn::VarStore::new(tch::Device::cuda_if_available());
                (net, stack) = if let Some((t1,t2)) = construct_model(&vs.root()){
                    (t1,t2)
                } else {
                    continue;
                };
                loaded_model = true;
                println!("Model successfully built.");
            }
            //train optimizer epochs
            "train" => {
                if input.len() != 3 {
                    println!("Wrong syntax. Type help train for more information.");
                    continue;
                };
                if !loaded_model {
                    println!("No loaded model, ignoring this command!");
                    continue;
                };
                let optim = input[1];
                let epochs = input[2].parse::<i64>()?;
                let mut optim = match optim {
                    "adam" => nn::Adam::default().build(&vs, 0.)?,
                    "sgd" => nn::Sgd::default().build(&vs, 0.)?,
                    _ => nn::Adam::default().build(&vs, 0.)?,
                };
                train_model(&vs.root(), &net, &mut optim, &data, epochs);
                println!("Finished Training! Maybe test accuracy now?");
            }
            //accuracy
            "accuracy" => {
                if input.len() != 1 {
                    println!("This command does not take any arguments, they will be ignored!");
                };
                if !loaded_model {
                    println!("No loaded model, ignoring this command!");
                    continue;
                };
                println!(
                    "The accuracy of the current model is {:.2}%",
                    accuracy_model(&net, &data, &vs.device()) * 100.
                );
            }
            //predict image
            "predict" => {
                if input.len() != 2 {
                    println!("Wrong syntax. Type help train for more information.");
                    continue;
                }
                if !loaded_model {
                    println!("No loaded model, ignoring this command!");
                    continue;
                }
                let image = input[1];
                let path = format!("test/{image}");
                let prediction = predict(&net, &path, &vs.device())?;
                println!("The model predicted: {}", prediction.as_str());
            }
            "view" => {
                if input.len() != 1 {
                    println!("This command does not take any arguments, they will be ignored!");
                };
                if !loaded_model {
                    println!("No loaded model, ignoring this command!");
                    continue;
                };
                println!("{}", stack);
            }
            //exit
            "exit" => {
                break;
            }
            _ => {
                println!("Invalid command, for help type \"help\"");
            }
        }
    }
    Ok(())
}

pub fn get_modelname(pathname: &str) -> String {
    let _t = pathname.to_string().split(".").collect::<Vec<&str>>()[0].to_string();
    let name = _t.split('/').collect::<Vec<&str>>()[1].clone();
    String::from(name)
}
pub fn from_stack(stack: &Layers, vs: &nn::Path) -> SequentialT {
    stack
        .0
        .iter()
        .fold(tch::nn::seq_t(), move |model, layer| match *layer {
            Layer::ConvLayer(in_channels, out_channels, kernel_size, stride, padding, bias) => {
                model.add(conv2d_sublayer(
                    vs,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias,
                ))
            }
            Layer::Maxpool(kernel) => model.add_fn(move |x| x.max_pool2d_default(kernel)),
            Layer::Dropout(dropout) => model.add_fn_t(move |x, train| x.dropout(dropout, train)),
            Layer::Flatten() => model.add_fn(|x| x.flat_view()),
            Layer::Linear(in_channels, out_channels) => model.add(tch::nn::linear(
                vs,
                in_channels,
                out_channels,
                Default::default(),
            )),
            Layer::Relu() => model.add_fn(|x| x.relu()),
            _ => model,
        })
}
impl Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &*self {
            Layer::ConvLayer(ic, oc, ks, s, p, b) => {
                write!(f, "ConvLayer({ic},{oc},{ks},{:?},{:?},{b})", s, p)
            }
            Layer::Maxpool(ks) => write!(f, "Maxpooling({ks})"),
            Layer::Dropout(p) => write!(f, "Dropout({p})"),
            Layer::Flatten() => write!(f, "Flatten"),
            Layer::Linear(ic, oc) => write!(f, "Linear({ic},{oc})"),
            Layer::Relu() => write!(f, "Relu"),
            Layer::Other(str) =>{
                match str.as_str(){
                    "cnn1" => write!(f,"ConvLayer(3,32,3,Some(1),Some(1),true)-> ConvLayer(32,64,3,None,Some(1),true)-> Maxpool(2)-> Dropout(0.25)-> ConvLayer(64,64,3,Some(1),Some(1),true)-> ConvLayer(64,128,3,Some(1),Some(1),true)-> Maxpool(2)-> Dropout(0.25)-> ConvLayer(128,128,3,Some(1),Some(1),true)-> ConvLayer(128,256,3,Some(1),Some(1),true)-> Maxpool(2)-> Dropout(0.25)-> Flatten-> Linear(4096,512)-> Linear(512,10)"),
                    "fastnet1" =>write!(f,"ConvLayer(3,64,3,None,Some(1),true)-> ConvLayer(64,256,3,None,Some(1),true) -> Maxpool(4)-> Dropout(0.25)-> ConvLayer(256,512,3,None,Some(1),true)-> Maxpool(2)-> Maxpool(4)-> Flatten-> Linear(512,10)"),
                    "fastnet2" =>write!(f,"ConvLayer(3,128,3,None,Some(1),true)-> ConvLayer(128,256,3,None,Some(1),true)-> Maxpool(4)-> Dropout(0.25)-> ConvLayer(256,256,3,None,Some(1),true)-> ConvLayer(256,512,3,None,Some(1),true)-> Maxpool(2)-> Dropout(0.25)-> Maxpool(4)->  Flatten-> Linear(512,10)" ),
                    _ =>write!(f,""),
                }
            } ,
        }
    }
}
impl Display for Layers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Input Tensor-> ")?;
        for layer in &self.0 {
            write!(f, "{layer}-> ").unwrap();
        }
        write!(f, "Output Tensor")
    }
}
