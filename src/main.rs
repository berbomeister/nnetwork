use nnetwork::*;
enum Layer {
    ConvLayer(i64, i64, i64, i64, i64, bool),
    Maxpool(i64),
    Dropout(f64),
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
            }
        } else if input[0] == "maxpool" {
            assert!(
                input.len() == 3,
                "{:?} - missing arguments",
                input.join(" ")
            );
            let kernel_size = i64::from_str_radix(input[2], 10).unwrap();
            stack.push(Layer::Maxpool(kernel_size));
        } else if input[0] == "end" {
            break;
        }
    }
    // while input[0] == "add" {
    //     if input[1] == "conv_layer" {
    //         assert!(input.len() == 6 || input.len() == 7);
    //         let in_channels = i64::from_str_radix(input[2], 10).unwrap();
    //         let out_channels = i64::from_str_radix(input[3], 10).unwrap();
    //         let kernel_size = i64::from_str_radix(input[4], 10).unwrap();
    //         let (stride, padding) = if input[5] == "--default" {
    //             (Some(1), Some(0))
    //         } else {
    //             (
    //                 Some(i64::from_str_radix(input[5], 10).unwrap()),
    //                 Some(i64::from_str_radix(input[6], 10).unwrap()),
    //             )
    //         };
    //         stack.push((
    //             in_channels,
    //             out_channels,
    //             kernel_size,
    //             stride,
    //             padding,
    //             true,
    //         ));
    //     }
    //     let _e = stdin.read_line(&mut user_input);
    //     input = &user_input.trim().split(" ").collect::<Vec<&str>>();
    // }
    let model = stack.iter().fold(tch::nn::seq_t(), move |model, layer| {
        match *layer {
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
            },
            Layer::Maxpool(kernel) => model.add_fn(move |x| x.max_pool2d_default(kernel)),
            Layer::Dropout(dropout) => model.add_fn_t(move |x,train| x.dropout(dropout, train)),
            Layer::Linear(in_channels, out_channels) => model.add(tch::nn::linear(&vs.root(), in_channels, out_channels, Default::default())),
        }
    });
    println!("{:?}", model);
    //add conv_layer [in channels] [out channels] [kernel_size] [--default |  [stride] [padding]]
    // let model_arch = user_input.as_str().trim().split(" ").collect::<Vec<&str>>();
    // println!("{:?}", model_arch);
    // for layer in model_arch {
    //     println!("{:?}",layer);

    // }
}
