use nnetwork::*;
#[derive(Debug)]
enum Layer {
    ConvLayer(i64, i64, i64, i64, i64, bool),
    Maxpool(i64),
    Dropout(f64),
    Flatten(),
    Linear(i64, i64),
}
pub fn main() -> () {
    // let _r = test();

    // let mut user_input = String::new();
    // let stdin = std::io::stdin();
    // let _e = stdin.read_line(&mut user_input);
    // let mut input = &user_input.trim().split(" ").collect::<Vec<&str>>();

    let vs = tch::nn::VarStore::new(tch::Device::cuda_if_available());
    let mut stack: Vec<Layer> = Vec::new();
    loop {
        let mut user_input = String::new();
        let stdin = std::io::stdin();
        let _e = stdin.read_line(&mut user_input);
        let input = &user_input.trim().split(" ").collect::<Vec<&str>>();
        if input[0] == "add" {
            assert!(
                input.len() >= 2,
                "{:?} - missing arguments",
                input.join(" ")
            );
            if input[1] == "conv_layer" {
                assert!(
                    input.len() == 6 || input.len() == 7,
                    "{:?} - missing arguments",
                    input.join(" ")
                );
                let in_channels = i64::from_str_radix(input[2], 10).unwrap();
                let out_channels = i64::from_str_radix(input[3], 10).unwrap();
                let kernel_size = i64::from_str_radix(input[4], 10).unwrap();
                let (stride, padding) = if input[5] == "--default" {
                    (1, 0)
                } else {
                    (
                        i64::from_str_radix(input[5], 10).unwrap(),
                        i64::from_str_radix(input[6], 10).unwrap(),
                    )
                };
                stack.push(Layer::ConvLayer(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    true,
                ));
            } else if input[1] == "maxpool" {
                assert!(
                    input.len() == 3,
                    "{:?} - missing arguments",
                    input.join(" ")
                );
                let kernel_size = i64::from_str_radix(input[2], 10).unwrap();
                stack.push(Layer::Maxpool(kernel_size));
            } else if input[1] == "dropout" {
                assert!(
                    input.len() == 3,
                    "{:?} - missing arguments",
                    input.join(" ")
                );
                let dropout = (input[2]).parse::<f64>().unwrap();
                stack.push(Layer::Dropout(dropout));
            } else if input[1] == "linear" {
                assert!(
                    input.len() == 4,
                    "{:?} - missing arguments",
                    input.join(" ")
                );
                let in_channels = (input[2]).parse::<i64>().unwrap();
                let out_channels = (input[3]).parse::<i64>().unwrap();
                stack.push(Layer::Linear(in_channels, out_channels));
            } else if input[1] == "flatten" {
                assert!(
                    input.len() == 2,
                    "{:?} - missing arguments",
                    input.join(" ")
                );
                stack.push(Layer::Flatten());
            } else {
                panic!("unknown layer name");
            }
        } else if input[0] == "end" {
            break;
        }
    }
    let model = stack
        .iter()
        .fold(tch::nn::seq_t(), move |model, layer| match *layer {
            Layer::ConvLayer(in_channels, out_channels, kernel_size, stride, padding, bias) => {
                model.add(conv2d_sublayer(
                    &vs.root(),
                    in_channels,
                    out_channels,
                    kernel_size,
                    Some(stride),
                    Some(padding),
                    bias,
                ))
            }
            Layer::Maxpool(kernel) => model.add_fn(move |x| x.max_pool2d_default(kernel)),
            Layer::Dropout(dropout) => model.add_fn_t(move |x, train| x.dropout(dropout, train)),
            Layer::Flatten() => model.add_fn(|x| x.flat_view()),
            Layer::Linear(in_channels, out_channels) => model.add(tch::nn::linear(
                &vs.root(),
                in_channels,
                out_channels,
                Default::default(),
            )),
        });
    println!("{:?}", model);
    println!("{:?}",stack);
    //add conv_layer [in channels] [out channels] [kernel_size] [--default |  [stride] [padding]]
    //add dropout dropout
    //add maxpool kernel_size
    //add flatten
    //add linear in_channels out_channels
}
