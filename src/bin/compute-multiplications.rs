// cargo run --bin compute-multiplications

use massey::*;

use anyhow::Result;
use dashmap::DashMap;

use fp::matrix::Matrix;

use adams::{AdamsGenerator, AdamsMultiplication, Bidegree};

fn callback(
    lhs: AdamsGenerator,
    _max_rhs_deg_computed: Bidegree,
    matrices: &DashMap<Bidegree, Matrix>,
) -> Result<(), String> {
    eprintln!("Multiplications computed for {}", lhs);
    for bideg in matrices.iter() {
        let bidegree = bideg.key();
        let matrix = bideg.value();
        for (idx, row) in matrix.iter().enumerate().filter(|(_, row)| !row.is_zero()) {
            let gen: AdamsGenerator = (*bidegree, idx).into();
            println!("{} * {} = {}", lhs, gen, row);
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let save_file_name = query::with_default(
        "Save directory",
        "../massey-prod-calc-data/S_2_resolution.data",
        |filename| core::result::Result::<_, std::convert::Infallible>::Ok(String::from(filename)),
    );

    let multiplication_data_directory = query::with_default(
        "Multiplication data directory",
        "../massey-prod-calc-data/S_2_multiplication_data",
        |filename| core::result::Result::<_, std::convert::Infallible>::Ok(String::from(filename)),
    );

    eprintln!("Loading resolution...");
    let mut adams_mult: AdamsMultiplication = AdamsMultiplication::new(
        save_file_name,
        None,
        Some(multiplication_data_directory),
        None,
        None,
    )?;

    eprintln!("Computing multiplications...");
    match adams_mult.compute_all_multiplications_callback(true, callback) {
        Ok(_) => {}
        Err(err_info) => {
            eprintln!("{}", err_info);
        }
    }

    Ok(())
}
