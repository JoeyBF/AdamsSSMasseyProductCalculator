// cargo run --bin compute-multiplications

use massey::*;

use anyhow::Result;
use dashmap::DashMap;

use fp::matrix::Matrix;

use adams::{AdamsGenerator, AdamsMultiplication, Bidegree};

fn callback(
    lhs: AdamsGenerator,
    _max_rhs_deg_computed: Bidegree,
    _matrices: &DashMap<Bidegree, Matrix>,
) -> Result<(), String> {
    println!("Multiplications computed for {}", lhs);
    Ok(())
}

fn main() -> Result<()> {
    println!("Loading resolution...");
    let mut adams_mult: AdamsMultiplication = AdamsMultiplication::new()?;

    println!("Computing multiplications...");
    match adams_mult.compute_all_multiplications_callback(true, callback) {
        Ok(_) => {}
        Err(err_info) => {
            eprintln!("{}", err_info);
        }
    }

    Ok(())
}
