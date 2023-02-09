
use anyhow::Result;
use tch::nn::{ ModuleT, OptimizerConfig};
use tch::{nn, Device};
use nnetwork::*;



pub fn main() -> Result<()> {
    let m = tch::vision::cifar::load_dir("data")?;
    let vs = nn::VarStore::new(Device::cuda_if_available());
    // let net = fast_resnet(&vs.root());
    // let net = cnn_net(&vs.root());
    let mut user_input = String::new();
    let stdin = std::io::stdin(); // We get `Stdin` here.
    stdin.read_line(&mut user_input)?;
    println!("{:?}",user_input.as_str().trim());
    let net = match user_input.as_str().trim() {
        "1" => fast_resnet(&vs.root()),
        _ => cnn_net(&vs.root()),
    };
    let mut opt = nn::Sgd {
        momentum: 0.9,
        dampening: 0.,
        wd: 5e-4,
        nesterov: true,
    }
    .build(&vs, 0.)?;
    for epoch in 1..150 {
        opt.set_lr(learning_rate(epoch));
        for (i, (bimages, blabels)) in m
            .train_iter(64)
            .shuffle()
            .to_device(vs.device())
            .enumerate()
        {
            let bimages = tch::vision::dataset::augmentation(&bimages, true, 4, 8);
            let pred = net.forward_t(&bimages, true);
            // println!("{}",pred);
            let loss = pred.cross_entropy_for_logits(&blabels);
            // println!("{}",loss);
            // let loss = net
            //     .forward_t(&bimages, true)
            //     .cross_entropy_for_logits(&blabels);
            opt.backward_step(&loss);
            if i % 100 == 0 {
                let test_accuracy =
                    net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 512);
                println!(
                    "epoch: {:4}, batch: {:5}, test acc: {:5.2}%",
                    epoch,
                    i,
                    100. * test_accuracy,
                );
            }
        }
        let test_accuracy =
            net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 512);
        println!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy,);
    }
    Ok(())
}
