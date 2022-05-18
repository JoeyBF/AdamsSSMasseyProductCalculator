// cargo run --bin verifier

// Verify massey products have expected additivity and multiplicativity properties

// import library root
use massey::*;

use anyhow::Result;

use std::{clone::Clone, cmp::Ordering};

use std::collections::hash_map::HashMap;

use fp::vector::FpVector;

use adams::{AdamsElement, AdamsMultiplication, Bidegree, MasseyProduct};

use affinespace::AffineSpace;

fn main() -> Result<()> {
    let max_s = 33;
    let max_t = 105;

    println!("Loading and extending resolution...");
    let mut adams_mult: AdamsMultiplication = AdamsMultiplication::new()?;
    let prime = adams_mult.prime();

    adams_mult.extend_resolution_to((max_s, max_t).into())?;

    //fp::vector::initialize_limb_bit_index_table(adams_mult.resolution().prime());

    println!("Loading and computing multiplications...");
    match adams_mult.compute_all_multiplications() {
        Ok(_) => {}
        Err(err_info) => {
            eprintln!("{}", err_info);
        }
    }

    let h0 = (1, 1, FpVector::from_slice(prime, &[1])).into();
    let h1: AdamsElement = (1, 2, FpVector::from_slice(prime, &[1])).into();
    //let max_massey_deg = (32,102).into();

    println!("Loading Massey products...");
    let massey_h1_h0: Vec<(AdamsElement, MasseyProduct)> = Vec::new();
    /*
    {
        let save_file = File::open(massey_product_save_file)?;
        let mut buf_save_file = BufReader::new(save_file);
        let _ = Bidegree::load(&mut buf_save_file, &())?;
        let n = usize::load(&mut buf_save_file, &())?;
        for _ in 0..n {
            let a = AdamsElement::load(&mut buf_save_file, &prime)?;
            let prod = MasseyProduct::load(&mut buf_save_file, &prime)?;
            massey_h1_h0.push((a, prod));
        }
    }
    */
    println!("{} products loaded", massey_h1_h0.len());
    // reorganize massey products by a's bidegree to make recognizing additive failure easier
    let mut massey_map: HashMap<Bidegree, HashMap<FpVector, MasseyProduct>> = HashMap::new();
    for (a, prod) in &massey_h1_h0 {
        match massey_map.get_mut(&a.degree()) {
            Some(hm) => {
                hm.insert(a.vec().clone(), prod.clone());
            }
            None => {
                massey_map.insert(a.degree(), [(a.vec().clone(), prod.clone())].into());
            }
        }
    }
    // check additivity
    println!("Checking additivity...");
    for (bidegree, prods) in &massey_map {
        let a_dim = adams_mult
            .num_gens(*bidegree)
            .expect("Bidegree should be computed");
        if a_dim > 1 {
            println!("For bidegree {} which has dimension {}...", bidegree, a_dim);

            for v1 in utils::AllVectorsIterator::new_whole_space(prime, a_dim) {
                let ae1 = (*bidegree, &v1).into();
                let zero1 = match adams_mult.zero_massey_product_for(&ae1, h1.degree(), &h0) {
                    Ok(p) => p,
                    Err(reason) => {
                        eprintln!(
                            "Couldn't generate zero massey product for ({},{}) because {}",
                            bidegree, v1, reason
                        );
                        continue;
                    }
                };
                let prod1 = prods.get(&v1).unwrap_or(&zero1);
                for v2 in utils::AllVectorsIterator::new_whole_space(prime, a_dim) {
                    let ae2 = (*bidegree, &v2).into();
                    let zero2 = match adams_mult.zero_massey_product_for(&ae2, h1.degree(), &h0) {
                        Ok(p) => p,
                        Err(reason) => {
                            eprintln!(
                                "Couldn't generate zero massey product for ({},{}) because {}",
                                bidegree, v2, reason
                            );
                            continue;
                        }
                    };
                    let prod2 = prods.get(&v2).unwrap_or(&zero2);
                    let mut v3 = v1.clone();
                    v3.add(&v2, 1);
                    let ae3 = (*bidegree, &v3).into();
                    let zero3 = match adams_mult.zero_massey_product_for(&ae3, h1.degree(), &h0) {
                        Ok(p) => p,
                        Err(reason) => {
                            eprintln!(
                                "Couldn't generate zero massey product for ({},{}) because {}",
                                bidegree, v3, reason
                            );
                            continue;
                        }
                    };
                    let prod3 = prods.get(&v3).unwrap_or(&zero3);
                    // have prod1, prod2, prod3
                    // check compatibility
                    let mut rep_sum = prod1.rep().clone();
                    rep_sum.add(prod2.rep(), 1);
                    let affine = AffineSpace::new(
                        rep_sum,
                        utils::subspace_sum(prod1.indet(), prod2.indet()),
                    );
                    match prod3.partial_cmp(&affine) {
                        None | Some(Ordering::Greater) => {
                            println!("Additivity fails for {} + {} = {}.", ae1, ae2, ae3);
                            println!("Have {} !<= {} + {}", prod3, prod1, prod2);
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    Ok(())
}
