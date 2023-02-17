use nnetwork::*;
use anyhow::{Ok, Result};
pub fn main() -> Result<()> {

    cli()?;


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