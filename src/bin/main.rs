// cargo run --bin main

use std::sync::Arc;

// import library root
use adams::AdamsMultiplication;
use massey::{
    adams::{AdamsGenerator, Bidegree},
    *,
};

use ext::chain_complex::{ChainComplex, ChainHomotopy};
use fp::vector::FpVector;

use rayon::prelude::*;

//pub mod computation;
//use computation::ComputationResult;

/* need to store the products
 * need to be able to extract Massey productable triples
 * then need to compute the Massey products and store them.
 * should be extensible
 * note Massey productable triples can involve non generators
 *
 * Multiplication is a bilinear map
 * Adams(s1,t1) x Adams(s2,t2) -> Adams(s1+s2,t1+t2)
 * Idea 1:
 * Store bilinear map as 3d matrix
 * (normal 2d matrix with entries in Adams(s1+s2,t1+t2))
 * Idea 2:
 * Store as linear map from the tensor product
 * Adams(s1,t1) \otimes Adams(s2,t2) -> Adams(s1+s2,t1+t2)
 * this is a normal matrix
 * Idea 3:
 * For each generator x_{s1,t1,i} in (s1, t1) store the matrix
 * for left multiplication x_{s1,t1,i}
 * (this is what we start with)
 * !!! We'll start with this !!! and adjust as necessary
 *
 * Goal is to compute pairs
 * (a,b) \in Adams(s1,t1)\times Adams(s2,t2) such that
 * mu(a,b) = 0
 *
 */

fn main() -> anyhow::Result<(), String> {
    println!("Loading resolution...");
    let mut adams_mult: AdamsMultiplication =
        AdamsMultiplication::new().map_err(|err| err.to_string())?;
    let p = adams_mult.prime();

    println!("Loading and computing multiplications...");
    adams_mult.compute_all_multiplications()?;

    adams_mult
        .resolution()
        .iter_stem()
        .par_bridge()
        .map(|(b_s, _, b_t)| -> Result<(), String> {
            let b_b = Bidegree::new(b_s, b_t);
            if b_b == Bidegree::new(0, 0) {
                return Ok(());
            }
            let b_num_gens = adams_mult.num_gens(b_b).unwrap_or_else(|| unreachable!());
            for b_idx in 0..b_num_gens {
                let b = AdamsGenerator::new(b_s, b_t, b_idx);
                let ref_b_mults = adams_mult
                    .multiplication_matrices()
                    .get(&b)
                    .unwrap_or_else(|| panic!("Multiplication not computed for generator {b}"));
                let (_, b_mults) = ref_b_mults.value();
                let b_hom = Arc::new(adams_mult.adams_gen_to_resoln_hom(b)?);
                println!("Computing resolution homomorphism for element b = {b}");
                b_hom.extend_all();
                adams_mult
                    .resolution()
                    .iter_stem()
                    .par_bridge()
                    .map(|(c_s, _, c_t)| -> Result<(), String> {
                        let c_b = Bidegree::new(c_s, c_t);
                        if c_b == Bidegree::new(0, 0) {
                            return Ok(());
                        }
                        let c_num_gens = adams_mult.num_gens(c_b).unwrap_or_else(|| unreachable!());
                        if c_num_gens == 0 {
                            eprintln!("Bidegree {c_b} empty, continuing...");
                            return Ok(());
                        }
                        let tot_b = b_b + c_b;
                        if !adams_mult
                            .resolution()
                            .has_computed_bidegree(tot_b.s(), tot_b.t())
                        {
                            eprintln!("Bidegree {tot_b} not computed, continuing...");
                            return Ok(());
                        }
                        let tot_num_gens = adams_mult.num_gens(tot_b).unwrap();
                        for c_idx in 0..c_num_gens {
                            let c = AdamsGenerator::new(c_s, c_t, c_idx);
                            let c_hom = Arc::new(adams_mult.adams_gen_to_resoln_hom(c)?);
                            println!("Computing resolution homomorphism for element c = {c}");
                            c_hom.extend_all();
                            if let Some(m) = b_mults.get(&c_b) {
                                let mut result = FpVector::new(p, tot_num_gens);
                                m.value().apply(
                                    result.as_slice_mut(),
                                    1,
                                    FpVector::from_slice(p, &c.vector(c_num_gens)).as_slice(),
                                );
                                if !result.is_zero() {
                                    eprintln!("Product {b} * {c} is nonzero, continuing...");
                                    continue;
                                }
                            }
                            println!("Computing nullhomotopy for pair ({b}, {c})");
                            let htpy = ChainHomotopy::new(Arc::clone(&c_hom), Arc::clone(&b_hom));
                            htpy.extend_all();
                        }
                        Ok(())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
            Ok(())
        })
        .collect::<Result<Vec<_>, _>>()?;
    // let h0 = (1, 1, FpVector::from_slice(p, &[1])).into();
    // let h1 = (1, 2, FpVector::from_slice(p, &[1])).into();

    // println!("Computing kernels for multiplication by h0 = {}...", h0);
    // // first compute kernels for h0
    // let kers_h0 = adams_mult.compute_kernels_right_multiplication(&h0)?;
    // println!("Computing massey products <-,{},{}>...", h0, h1);
    // let massey_h0_h1 = adams_mult.compute_massey_prods_for_pair(&kers_h0, &h0, &h1);
    // println!("Massey products <-,{},{}> computed", h0, h1);
    // let shift_deg = (1, 3).into();
    // for (a, rep) in massey_h0_h1 {
    //     let rep_ae: AdamsElement = (a.degree() + shift_deg, rep.rep()).into();
    //     println!("<{a}, b, c> = {rep_ae}");
    // }

    //println!("Computing kernels for multiplication by h1 = {}...", h1);
    // // first compute kernels for h0
    // let kers_h1 = match adams_mult.compute_kernels_right_multiplication(&h1) {
    //     Ok(kers) => kers,
    //     Err(err_info) => {
    //         println!("{}", err_info);
    //         // fail
    //         return Err(err_info);
    //         //std::process::exit(-1);
    //     }
    // };
    //println!("Computing massey products <-,{},{}>...", h1, h0);
    // let massey_h1_h0 = adams_mult.compute_massey_prods_for_pair(&kers_h1, &h1, &h0);
    //println!("Massey products <-,{},{}> computed", h1, h0);
    // let shift_deg = (1, 3).into();
    // for (a, rep) in massey_h1_h0 {
    //     let rep_ae: AdamsElement = (a.degree() + shift_deg, rep.rep()).into();
    //    println!("<a, b, {}> = {}", a, rep_ae);
    // }

    //adams_mult.possible_nontrivial_massey_products();

    Ok(())
}
