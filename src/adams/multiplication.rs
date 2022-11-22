use anyhow;
use dashmap::DashMap;
use rayon::prelude::*;

use std::clone::Clone;
use std::cmp::Ordering;
use std::io;
use std::sync::Arc;

use algebra::module::FDModule;
use algebra::MilnorAlgebra;

use ext::chain_complex::{ChainComplex, ChainHomotopy, FreeChainComplex};
use ext::nassau::Resolution as NassauResolution;
use ext::resolution_homomorphism::ResolutionHomomorphism;
use ext::utils::query_module;

use fp::matrix::Matrix;
use fp::matrix::Subspace;
use fp::prime::ValidPrime;
use fp::vector::FpVector;

use super::{AdamsElement, AdamsGenerator, Bidegree, MasseyProduct};

use crate::lattice::{meet, JoinSemilattice, MeetSemilattice};
use crate::utils::AllVectorsIterator;

pub type Resolution = NassauResolution<FDModule<MilnorAlgebra>>;

//#[derive(Clone)]
pub struct AdamsMultiplication {
    /// the resolution object
    resolution: Arc<Resolution>,
    /// max bidegree w/ dimensions computed
    max_bidegree: Bidegree,
    /// stores the multiplication matrices for each degree
    /// where we could compute the multiplication
    multiplication_matrices: DashMap<AdamsGenerator, (Bidegree, DashMap<Bidegree, Matrix>)>,
}

impl AdamsMultiplication {
    pub fn max_deg(&self) -> Bidegree {
        self.max_bidegree
    }

    pub fn extend_resolution_to(&mut self, deg: Bidegree) -> io::Result<()> {
        if deg <= self.max_deg() {
            return Ok(());
        }

        let mut cur_deg = self.max_deg();

        while cur_deg < deg {
            if cur_deg.s() < deg.s() {
                *cur_deg.s_mut() += 1;
            } else {
                *cur_deg.t_mut() += 1;
            }

            let (s, t) = cur_deg.into();
            eprintln!("Extending to degree {}", cur_deg);

            self.resolution.compute_through_bidegree(s, t);

            /*
            if self.has_resolution_data_directory() {
                let file: File = File::create(self.resolution_file_path(cur_deg).expect("unexpectedly resolution file path was None"))?;
                let mut buf_file = std::io::BufWriter::new(file);
                self.resolution.save(&mut buf_file)?;
            }
            */
        }
        /*
        let file: File = File::create(&self.res_file_name)?;
        let mut buf_file = std::io::BufWriter::new(file);
        self.resolution.save(&mut buf_file)?;
        */
        // update max degree
        *self.max_bidegree.s_mut() = deg.s;
        *self.max_bidegree.t_mut() = deg.t;
        Ok(())
    }

    pub fn new() -> anyhow::Result<AdamsMultiplication> {
        let res = Arc::new(query_module(None, false)?);
        let max_bidegree = res
            .iter_stem()
            .last()
            .map(|(s, _, t)| Bidegree::new(s, t))
            .unwrap();

        Ok(AdamsMultiplication {
            resolution: res,
            max_bidegree,
            multiplication_matrices: DashMap::new(),
        })
    }

    /// return Arc to the resolution
    pub fn resolution(&self) -> Arc<Resolution> {
        Arc::clone(&self.resolution)
    }

    pub fn prime(&self) -> ValidPrime {
        self.resolution.prime()
    }

    /// return nonmutable reference
    pub fn multiplication_matrices(
        &self,
    ) -> &DashMap<AdamsGenerator, (Bidegree, DashMap<Bidegree, Matrix>)> {
        &self.multiplication_matrices
    }

    pub fn num_gens<T: Into<(u32, i32)>>(&self, bidegree: T) -> Option<usize> {
        let (s, t) = bidegree.into();
        if self.resolution.has_computed_bidegree(s, t) {
            Some(self.resolution.number_of_gens_in_bidegree(s, t))
        } else {
            None
        }
    }

    pub fn adams_gen_to_resoln_hom(
        &self,
        g: AdamsGenerator,
    ) -> Result<ResolutionHomomorphism<Resolution, Resolution>, String> {
        let (s, t, idx) = g.into();
        let dim = self
            .num_gens((s, t))
            .ok_or(format!("resolution not computed through ({}, {})", s, t))?;
        Ok(ResolutionHomomorphism::from_class(
            format!("({},{},{})", s, t, idx),
            self.resolution(),
            self.resolution(),
            s,
            t,
            &g.vector(dim),
        ))
    }

    pub fn adams_elt_to_resoln_hom(
        &self,
        e: &AdamsElement,
    ) -> ResolutionHomomorphism<Resolution, Resolution> {
        if let Ok(g) = e.try_into() {
            return self.adams_gen_to_resoln_hom(g).unwrap();
        }
        let (s, t, v) = e.into();
        let vec = v.iter().collect::<Vec<_>>();
        ResolutionHomomorphism::from_class(
            format!("({},{},{})", s, t, v),
            self.resolution(),
            self.resolution(),
            s,
            t,
            &vec,
        )
    }

    pub fn compute_multiplication(
        &self,
        g: AdamsGenerator,
        mult_with_max: Bidegree,
    ) -> Result<(Bidegree, DashMap<Bidegree, Matrix>), String> {
        eprintln!("Computing multiplication for {}", g);
        let (s, t, _) = g.into();
        let g_deg = g.degree();
        let max_deg = self.max_deg();
        match g_deg.partial_cmp(&max_deg) {
            None | Some(Ordering::Greater) => {
                return Err(format!(
                    "{g} is out of computed range: {}",
                    self.max_bidegree
                ));
            }
            _ => {}
        };

        let rmax_poss = self
            .max_bidegree
            .try_subtract((s, t))
            .unwrap_or_else(|| unreachable!());

        let actual_rmax = mult_with_max.meet(rmax_poss);

        let (partially_computed, computed_range, hm) = (false, (0, 0).into(), DashMap::new());

        let compute_to = if partially_computed {
            // determine what's left to compute
            if actual_rmax <= computed_range {
                // done, can't compute more
                return Ok((computed_range, hm));
            }
            // this should be true, or we're going to run into problems
            // might remove this restriction later
            assert!(rmax_poss >= computed_range);
            computed_range.join(actual_rmax) // should compute a strictly larger rectangle
        } else {
            actual_rmax
        };
        let hom = self.adams_gen_to_resoln_hom(g)?;

        // extend hom

        hom.extend_all();

        // then read off and insert the multiplication matrices

        // ok let's do the proper multiplications
        for (sasdf, _, tasdf) in self.resolution().iter_stem() {
            let rhs = Bidegree::new(sasdf, tasdf);
            // might want to iterate over stems, since these are the only nonzero ones, but for now
            // we just skip those with n negative
            if rhs.n() < 0 {
                continue; // rhs trivially empty, since n is negative
            }
            if partially_computed && rhs <= computed_range {
                continue; // already computed, no need to compute again
            }
            let target_deg: Bidegree = (s + rhs.s(), t + rhs.t()).into();
            let dim_rhs = match self.num_gens(rhs) {
                Some(n) => n,
                None => {
                    continue;
                    // return Err(format!(
                    //     "Dimension at rhs {} not computed. Expected for computing multiplication by {} in degree {}",
                    //     rhs,
                    //     g,
                    //     rhs,
                    // ));
                }
            };
            let dim_target = match self.num_gens(target_deg) {
                Some(n) => n,
                None => {
                    continue;
                    // return Err(format!(
                    //     "Dimension at target {} not computed. Expected for computing multiplication by {} in degree {}",
                    //     target_deg,
                    //     g,
                    //     rhs,
                    // ));
                } // this is an error
            };
            if dim_rhs == 0 || dim_target == 0 {
                continue; // nothing in domain, or nothing in codomain, multiplication is trivially 0
                          // store nothing
            }
            //let gens2 = &module2.gen_names()[j2];
            let matrix = hom.get_map(target_deg.s()).hom_k(rhs.t());
            // convert to fp::matrix::Matrix and store
            hm.insert(rhs, Matrix::from_vec(self.prime(), &matrix));
        }

        Ok((compute_to, hm))
    }

    pub fn compute_all_multiplications(&mut self) -> Result<(), String> {
        //self.compute_multiplications(self.max_s, self.max_t, self.max_s, self.max_t);
        self.compute_multiplications(self.max_deg(), self.max_deg())
    }

    pub fn compute_multiplications(
        &mut self,
        lhs_max: Bidegree,
        rhs_max: Bidegree,
    ) -> Result<(), String> {
        let lhs_max = lhs_max.meet(self.max_deg()); // don't go out of range
        let rhs_max = rhs_max.meet(self.max_deg()); // don't go out of range
        for lhs in lhs_max.iter_stem() {
            let dim = self.num_gens(lhs).ok_or_else(|| {
                format!("compute_multiplications: Expected {lhs} to be computed!")
            })?;
            for idx in 0..dim {
                let g = (lhs, idx).into();
                let mult_data = self.compute_multiplication(g, rhs_max)?;
                self.multiplication_matrices.insert(g, mult_data);
            }
        }
        Ok(())
    }

    // / boolean variable store determines whether or not to store the multiplication matrices
    // / callback called with adams generator, max bidegree for which multiplication matrices were computed
    // / and the hashmap of multiplication matrices for multiplication by the adams generator
    pub fn compute_all_multiplications_callback<F>(
        &mut self,
        store: bool,
        callback: F,
    ) -> Result<(), String>
    where
        F: Fn(AdamsGenerator, Bidegree, &DashMap<Bidegree, Matrix>) -> Result<(), String>
            + Send
            + Sync,
    {
        self.compute_multiplications_callback(self.max_deg(), self.max_deg(), store, callback)
    }

    /// boolean variable store determines whether or not to store the multiplication matrices
    /// callback called with adams generator, max bidegree for which multiplication matrices were computed
    /// and the hashmap of multiplication matrices for multiplication by the adams generator
    pub fn compute_multiplications_callback<F>(
        &mut self,
        lhs_max: Bidegree,
        rhs_max: Bidegree,
        store: bool,
        callback: F,
    ) -> Result<(), String>
    where
        F: Fn(AdamsGenerator, Bidegree, &DashMap<Bidegree, Matrix>) -> Result<(), String>
            + Send
            + Sync,
    {
        let lhs_max = lhs_max.meet(self.max_deg()); // don't go out of range
        let rhs_max = rhs_max.meet(self.max_deg()); // don't go out of range
        let f = Arc::new(callback);
        lhs_max
            .iter_stem()
            .par_bridge()
            .map(|lhs| {
                let dim = self.num_gens(lhs).ok_or_else(|| {
                    format!("compute_multiplications: Expected {lhs} to be computed!")
                })?;
                (0..dim)
                    .into_par_iter()
                    .map(|idx| {
                        let g = (lhs, idx).into();
                        let mult_data = self.compute_multiplication(g, rhs_max)?;
                        f(g, mult_data.0, &mult_data.1)?;
                        if store {
                            self.multiplication_matrices.insert(g, mult_data);
                        }
                        Result::<(), String>::Ok(())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(())
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|_| ())
    }

    /// only uses the dimensions, none of the multiplicative structure
    pub fn possible_nontrivial_massey_products(&self) {
        let mut count = 0;
        let mut not_all_dim_1 = 0;
        let mut max_triples = 0;
        for bideg1 in self.max_bidegree.iter_stem() {
            let n1 = match self.num_gens(bideg1) {
                Some(0) | None => continue,
                Some(n) => n,
            };
            for bideg2 in self.max_bidegree.iter_stem() {
                let n2 = match self.num_gens(bideg2) {
                    Some(0) | None => continue,
                    Some(n) => n,
                };

                let (shift_s, _) = (bideg1 + bideg2)
                    .try_subtract((1, 0))
                    .unwrap_or_else(|| unreachable!())
                    .into();

                if shift_s > self.max_bidegree.s() {
                    continue;
                }

                for bideg3 in self.max_bidegree.iter_stem() {
                    let n3 = match self.num_gens(bideg3) {
                        Some(0) | None => continue,
                        Some(n) => n,
                    };

                    let target = Bidegree::massey_bidegree(bideg1, bideg2, bideg3);
                    let target_n = match self.num_gens(target) {
                        Some(0) | None => continue,
                        Some(n) => n,
                    };

                    if n1 * n2 * n3 > 1 {
                        eprintln!(
                            "Potential massey products {bideg1} x {bideg2} x {bideg3} -> {target}"
                        );
                        eprintln!("Dimensions: {n1} x {n2} x {n3} -> {target_n}");
                        not_all_dim_1 += 1;
                    }
                    count += 1;
                    max_triples += n1 * n2 * n3;
                }
            }
        }

        eprintln!("VS triples: {}", count);
        eprintln!("VS triples not all linear: {}", not_all_dim_1);
        eprintln!("Max elt triples: {}", max_triples);
    }

    pub fn left_multiplication_by(
        &self,
        l: Bidegree,
        vec: &FpVector,
        r: Bidegree,
    ) -> Result<Matrix, String> {
        let p = self.prime();
        let g = l + r;

        let dim_l = self
            .num_gens(l)
            .ok_or(format!("Couldn't get generators in left bidegree {l}"))?;

        assert_eq!(dim_l, vec.len()); // vec has to have same length as number of gens in bidegree l

        let dim_r = self
            .num_gens(r)
            .ok_or(format!("Couldn't get generators in right bidegree {r}"))?;

        let dim_g = self
            .num_gens(g)
            .ok_or(format!("Couldn't get generators in target bidegree {g}"))?;

        let mut result = Matrix::new(p, dim_r, dim_g);
        if (dim_l == 0) || (dim_r == 0) || (dim_g == 0) {
            Ok(result)
        } else {
            for l_idx in 0..dim_l {
                let l_gen = (l, l_idx).into();
                let coeff = vec.entry(l_idx);
                if coeff == 0 {
                    continue;
                }
                let matrix = match self.multiplication_matrices.get(&l_gen) {
                    Some(kv) => {
                        let hm = &kv.1;
                        match hm.get(&r) {
                            Some(m) => m.value().clone(),
                            None => {
                                return Err(format!("Couldn't get multiplication matrix for gen {l_gen} in bidegree {r}"));
                            }
                        }
                    }
                    None => {
                        return Err(format!(
                            "Couldn't get multiplication matrices for gen {l_gen}"
                        )); // couldn't find an important multiplication matrix
                    }
                };
                result += /* coeff* */ &matrix; // coeff is 1 though, so we're good
            }
            Ok(result)
        }
    }

    pub fn right_multiplication_by(
        &self,
        r: Bidegree,
        vec: &FpVector,
        l: Bidegree,
    ) -> Result<Matrix, String> {
        // TODO right now I'm assuming that the multiplication in the Adams Spectral Sequence is commutative
        // so we can return
        self.left_multiplication_by(r, vec, l)
        // Check this assumption
    }

    /// Indeterminacy of massey product only depends on the bidegree of the middle term.
    /// returns the pair of subspaces of  
    /// aR, Rc in R
    /// in the bidegree where <a,b,c> lives
    pub fn compute_indeterminacy_of_massey_product(
        &self,
        a: &AdamsElement,
        b: Bidegree,
        c: &AdamsElement,
    ) -> Result<(Subspace, Subspace), String> {
        let (a_bidegree, v1) = a.into();
        let (c_bidegree, v3) = c.into();

        let left = (a_bidegree + b).try_subtract((1, 0)).unwrap();
        let right = (b + c_bidegree).try_subtract((1, 0)).unwrap();
        let total = Bidegree::massey_bidegree(a_bidegree, b, c_bidegree);

        let p = self.prime();

        let left_dim = self
            .num_gens(left)
            .ok_or(format!("Couldn't get dimension of {left}"))?;

        let right_dim = self
            .num_gens(right)
            .ok_or(format!("Couldn't get dimension of {right}"))?;

        let total_dim = self
            .num_gens(total)
            .ok_or(format!("Couldn't get dimension of {total}"))?;

        let l_indet = if right_dim == 0 {
            Subspace::empty_space(p, total_dim)
        } else {
            // compute left multiplication
            // from (s_right, t_right) -> (s_tot, t_tot)
            let left_indet = self
                .left_multiplication_by(a_bidegree, v1, right)
                .map_err(|err| {
                    format!("Couldn't compute the left multiplication {a}* : {right} -> {total} because {err}")
                })?;

            let (l_aug_start, mut l_indet_aug) =
                Matrix::augmented_from_vec(p, &left_indet.to_vec());
            l_indet_aug.row_reduce();

            l_indet_aug.compute_image(left_indet.columns(), l_aug_start)
        };

        let r_indet = if left_dim == 0 {
            Subspace::empty_space(p, total_dim)
        } else {
            // compute left multiplication
            // from (s_right, t_right) -> (s_tot, t_tot)
            let right_indet = self
                .right_multiplication_by(c_bidegree, v3, left)
                .map_err(|err| {
                    format!("Couldn't compute the right multiplication *{c} : {left} -> {total} because {err}")
                })?;

            let (r_aug_start, mut r_indet_aug) =
                Matrix::augmented_from_vec(p, &right_indet.to_vec());
            r_indet_aug.row_reduce();

            r_indet_aug.compute_image(right_indet.columns(), r_aug_start)
        };

        Ok((l_indet, r_indet))
    }

    pub fn zero_massey_product_for(
        &self,
        a: &AdamsElement,
        b: Bidegree,
        c: &AdamsElement,
    ) -> Result<MasseyProduct, String> {
        let (l_ind, r_ind) = self.compute_indeterminacy_of_massey_product(a, b, c)?;
        let bidegree = Bidegree::massey_bidegree(a.degree(), b, c.degree());
        let res_dim = self
            .num_gens(bidegree)
            .map(Ok)
            .unwrap_or(Err("massey product resulting group not computed"))?;
        Ok(MasseyProduct::new_ae(
            &(bidegree, FpVector::new(self.prime(), res_dim)).into(),
            l_ind,
            r_ind,
        ))
    }

    /// computes the maximum degree through which multiplication with an adams element is defined
    /// expects the adams generator to be valid
    pub fn multiplication_computed_through_degree(&self, gen: AdamsGenerator) -> Option<Bidegree> {
        self.multiplication_matrices.get(&gen).map(|kv| kv.0)
    }

    /// returns maximum degree such that all multiplications with generators living in multiplier_deg
    /// are defined through that degree. Assumes multiplier_deg <= self.max_deg()
    /// returns None if any prerequisites are not computed
    pub fn multiplication_completely_computed_through_degree(
        &self,
        multiplier_deg: Bidegree,
    ) -> Option<Bidegree> {
        let dim = match self.num_gens(multiplier_deg) {
            Some(n) => n,
            None => {
                // bidegree dimension not even computed
                return None;
            }
        };
        // max possible degree if all multiplications are computed
        let mut res = self.max_bidegree.try_subtract(multiplier_deg).unwrap();
        for idx in 0..dim {
            let max_for_idx =
                match self.multiplication_computed_through_degree((multiplier_deg, idx).into()) {
                    Some(deg) => deg,
                    None => {
                        // multiplication with this generator not computed at all
                        return None;
                    }
                };
            res = meet(res, max_for_idx);
        }
        Some(res)
    }

    /// compute kernels for left multiplication by the given AdamsElement through the given Bidegree
    pub fn compute_kernels_left_multiplication(
        &self,
        multiplier: &AdamsElement,
    ) -> Result<(Bidegree, DashMap<Bidegree, Subspace>), String> {
        // eprintln!(
        //     "compute_kernels_left_multiplication({}, {})",
        //     multiplier, max_degree_kernels
        // );
        let max_degree_kernels = self.max_deg();
        let deg1 = multiplier.degree();
        let hm = DashMap::new();
        let max_mult_with_deg = self
            .multiplication_completely_computed_through_degree(deg1)
            .ok_or_else(|| format!("Multiplication not completely computed for bidegree {deg1}"))?;
        let max_deg_for_kernels = meet(max_degree_kernels, max_mult_with_deg);
        for deg2 in max_deg_for_kernels.iter_s_t() {
            let deg3 = deg1 + deg2;
            let dim2 = match self.num_gens(deg2) {
                Some(x) => x,
                None => {
                    eprintln!(
                        "Trying to compute kernel for {multiplier}.
                        Expected multiply with degree {deg2} to be computed",
                    );
                    continue;
                }
            };
            if dim2 == 0 {
                continue; // no nonzero vectors, kernel is trivial
            }
            let dim3 = match self.num_gens(deg3) {
                Some(x) => x,
                None => {
                    eprintln!(
                        "Trying to compute kernel for {multiplier}.
                        Expected product with degree {deg3} to be computed",
                    );
                    continue;
                }
            };
            if dim3 == 0 {
                // kernel is everything
                // add and skip
                hm.insert(deg2, Subspace::entire_space(self.prime(), dim2));
                continue;
            }
            eprintln!("computing nontrivial kernel for deg: {}", deg2);
            let lmul_v1 = match self.left_multiplication_by(deg1, multiplier.vec(), deg2) {
                Ok(m) => m,
                Err(_) => {
                    continue;
                }
            };
            let (aug_start, mut lmul_v1_aug) =
                Matrix::augmented_from_vec(self.prime(), &lmul_v1.to_vec());
            lmul_v1_aug.row_reduce();
            let kernel_lmul_v1 = lmul_v1_aug.compute_kernel(aug_start);
            if kernel_lmul_v1.dimension() == 0 {
                // kernel trival
                continue; // skip
            }
            hm.insert(deg2, kernel_lmul_v1);
        }
        Ok((max_deg_for_kernels, hm))
    }

    /// computes kernels for right multiplication by the given Adams element through the given bidegree
    /// redirects to compute_kernels_left_multiplication right now. TODO
    pub fn compute_kernels_right_multiplication(
        &self,
        multiplier: &AdamsElement,
    ) -> Result<(Bidegree, DashMap<Bidegree, Subspace>), String> {
        // eprintln!(
        //     "compute_kernels_right_multiplication({}, {})",
        //     multiplier, max_degree_kernels
        // );
        self.compute_kernels_left_multiplication(multiplier)
    }

    /// computes kernels for left multiplication by all Adams elements through
    /// bidegree max_degree_multiplier with kernels computed through degree
    /// max_degree_kernel.
    /// returns hashmap from Adams elements to pairs of a bidegree representing the maximum degree through which
    /// kernels are computed and a hashmap with kernels by Bidegree
    /// if an in range pair of (nonzero) adams element and bidegree is not in the hashmaps,
    /// it's because the kernel is 0
    pub fn compute_all_kernels_left_multiplication(
        &self,
        max_degree_multiplier: Bidegree,
    ) -> Result<DashMap<AdamsElement, (Bidegree, DashMap<Bidegree, Subspace>)>, String> {
        let kernels: DashMap<AdamsElement, (Bidegree, DashMap<Bidegree, Subspace>)> =
            DashMap::new();
        for deg1 in max_degree_multiplier.iter_s_t() {
            let dim1 = match self.num_gens(deg1) {
                Some(n) => n,
                None => {
                    return Err(format!("Bidegree {} not computed", deg1));
                }
            };
            if dim1 == 0 {
                continue;
            }
            for v in AllVectorsIterator::new_whole_space(self.prime(), dim1) {
                let ae = AdamsElement::from((deg1, v));
                let kers = self.compute_kernels_left_multiplication(&ae)?;

                kernels.insert(ae, kers);
            }
        }
        Ok(kernels)
    }
    /// computes kernels for right multiplication by all Adams elements through
    /// bidegree max_degree_multiplier with kernels computed through degree
    /// max_degree_kernel
    ///
    /// currently implemented by just calling the compute_kernels_left_multiplication method
    pub fn compute_all_kernels_right_multiplication(
        &self,
        max_degree_multiplier: Bidegree,
    ) -> Result<DashMap<AdamsElement, (Bidegree, DashMap<Bidegree, Subspace>)>, String> {
        // TODO right now we're going to assume multiplication is commutative, since we're working with
        // Adams SS for sphere at p=2.
        self.compute_all_kernels_left_multiplication(max_degree_multiplier)
    }

    /// Write a new function to compute the massey products <a,b,c> given
    /// b and c for a less than max_deg_a.
    ///
    /// Should only compute one homotopy. Assumes bc=0.
    ///
    /// Returns the maximum a degree through which Massey products were computed, as well as
    /// a Vector of triples (a, representative, subspace) for which the massey product does not
    /// contain 0. Any possible massey product in the range indicated by the bidegree which is not
    /// recorded contains 0.
    ///
    /// takes the kernels for right multiplication by b as an argument
    pub fn compute_massey_prods_for_pair(
        &self,
        kernels_mul_b: &(Bidegree, DashMap<Bidegree, Subspace>),
        b: &AdamsElement,
        c: &AdamsElement,
    ) -> Vec<(AdamsElement, MasseyProduct)> {
        // eprintln!(
        //     "compute_massey_prods_for_pair(kernels, {}, {}, {})",
        //     max_deg_a, b, c
        // );
        // let max_deg_a = self.max_deg();
        let mut ans = vec![];
        let (_, ker_map) = kernels_mul_b;
        let b_deg = b.degree();
        let c_deg = c.degree();
        let b_c_deg = b_deg + c_deg;
        let b_c_shift = match b_c_deg.try_subtract((1, 0)) {
            Some(shift) => shift, // this is the degree difference |<a,b,c>| - |a|
            None => {
                return ans; // return empty data structure.
                            // this only happens if b and c have s degree 0, and therefore are either 0 or 1,
                            // which have no interesting massey products
            }
        };
        // // the extra 1 in s degree is due to massey product living in degree deg_a + deg_b + deg_c - (1,0)
        // let complement = match (self.max_deg() + (1, 0).into()).try_subtract(b_c_deg) {
        //     Some(cmpl) => cmpl,
        //     None => {
        //         return ans; // return empty data structure, since there are no valid a's to multiply with anyway
        //     }
        // };
        // // complement represents maximum possible a degree we can compute with
        // let max_a = max_deg_a.meet(complement).meet(*kernels_max_deg); // intersect the ranges
        // eprintln!("determined max_a: (s,t)=({},{})", max_a.s(), max_a.t());

        // let (max_s1, max_t1) = max_a.into();
        // let (s2, t2, v2) = b.into();
        // let (s3, t3, v3) = c.into();
        // // largest degree increase from c to <a,b,c>
        // let (max_shift_s, max_shift_t) = (max_s1 + s2 - 1, max_t1 + t2);
        // eprintln!("max_shift: (s,t)=({},{})", max_shift_s, max_shift_t);
        // //let shift_n = shift_t-shift_s as i32;
        // // largest total degree of <a,b,c>
        // let (max_tot_s, max_tot_t) = (max_shift_s + s3, max_shift_t + t3);
        // eprintln!("max_tot: (s,t)=({},{})", max_tot_s, max_tot_t);

        //let tot_n = tot_t-tot_s as i32;

        // for now we'll just compute the resolutions for b and c
        // this can be computed in terms of cached data in principle, and it should be faster,
        // but it requires a change to Hood's library
        eprint!("lifting {} to resolution homomorphism...", b);
        let b_hom = self.adams_elt_to_resoln_hom(b);
        b_hom.extend_all();
        eprintln!(" done.");
        eprint!("lifting {} to resolution homomorphism...", c);
        let c_hom = self.adams_elt_to_resoln_hom(c);
        c_hom.extend_all();
        eprintln!(" done.");
        eprint!("computing nullhomotopy of {} o {}...", b, c);
        let homotopy = ChainHomotopy::new(Arc::new(c_hom), Arc::new(b_hom));
        homotopy.extend_all();
        eprintln!(" done.");

        eprintln!("extracting massey products from homotopy...");
        // extract representatives for massey products from homotopy
        for (s, _, t) in self.resolution().iter_stem() {
            let a_deg = Bidegree::new(s, t);
            if a_deg.n() < 0 {
                continue;
            }
            let ker_b_dim_a = match ker_map.get(&a_deg) {
                Some(subsp) => {
                    if subsp.dimension() == 0 {
                        eprintln!("no vectors in kernel here, done. But kernel is recorded, which shouldn't happen.");
                        continue; // kernel is trivial, nothing interesting here.
                    }
                    subsp
                }
                None => {
                    //eprintln!("no vectors in kernel here, done.");
                    continue; // means that the kernel is trivial, nothing interesting here
                }
            };
            let (s1, t1) = a_deg.into();
            let tot_deg = a_deg + b_c_shift;
            let (tot_s, tot_t) = tot_deg.into();
            let target_dim = match self.num_gens(tot_deg) {
                Some(n) => n,
                None => {
                    eprintln!("Error, expected dimension of target for massey product <{},{},{}> to be computed", a_deg, b, c);
                    continue; // ignore TODO
                }
            };
            if target_dim == 0 {
                //eprintln!("target empty, done.");
                continue;
            }
            eprint!("for deg {}... ", a_deg);
            let htpy_map = homotopy.homotopy(tot_s);
            let offset_a = self.resolution.module(s1).generator_offset(t1, t1, 0); // where do generators
                                                                                   // start in the basis after all the products and what
            for vec_a in AllVectorsIterator::new(&ker_b_dim_a) {
                if vec_a.is_zero() {
                    continue; // trivial massey products
                }
                // now htpy_map
                // goes from the free module in degree tot_s to the one in degree a_deg
                // need to compose with the map given by vec_a from a_deg to F2
                // and then read off the values of the composite map
                // on the generators in degree tot_s, tot_t
                let mut answer = vec![0; target_dim];
                let mut nonzero = false;
                eprintln!(" computing for {}...", vec_a);
                for (i, ans) in answer.iter_mut().enumerate().take(target_dim) {
                    let output = htpy_map.output(tot_t, i);
                    // eprintln!(
                    //     "output of htpy for ({}, {}) index {} = {}",
                    //     tot_s, tot_t, i, output
                    // );
                    for (k, entry) in vec_a.iter().enumerate() {
                        if entry != 0 {
                            //answer[i] += entry * output.entry(self.resolution.module(s1).generator_offset(t1,t1,k));
                            *ans += entry * output.entry(offset_a + k);
                        }
                    }
                    if *ans != 0 {
                        nonzero = true;
                    }
                }
                // eprintln!(" rep for <{},-,->={:?}", vec_a, answer);
                if nonzero {
                    let massey_rep = FpVector::from_slice(self.prime(), &answer);
                    // eprintln!(" nonzero rep for <{},-,->={}", vec_a, massey_rep);
                    let ae1 = (a_deg, &vec_a).into();
                    let indets = match self.compute_indeterminacy_of_massey_product(&ae1, b_deg, c)
                    {
                        Ok(indets) => indets,
                        Err(reason) => {
                            eprintln!(
                                "< ({s1}, {t1}, {vec_a}), {b}, {c} > =
                                ({tot_s}, {tot_t}, {massey_rep}) + {{??}} could not compute indeterminacy because {reason}"
                            );
                            // hopefully this doesn't happen
                            continue; // printed out, keep on going
                        }
                    };
                    let massey_prod =
                        MasseyProduct::new(tot_s, tot_t, massey_rep, indets.0, indets.1);
                    if massey_prod.contains_zero() {
                        continue; // massey product is trivial, ignore it
                    }
                    ans.push((ae1, massey_prod))
                }
            }
            eprintln!("done.");
        }
        ans
    }

    /*
    pub fn compute_massey_products(&self, max_massey: Bidegree) {

    }
    */

    /*
    /// compute all massey products of massey-productable triples (a,b,c)
    /// all of whose bidegrees are less than max_massey
    pub fn brute_force_compute_all_massey_products(&self, max_massey: Bidegree) {
        let mut zero_massey_output = match File::create("zero-massey-prods.txt") {
            Err(error) => { eprintln!("Could not open 'zero-massey-prods.txt' for writing: {}", error); return; }
            Ok(file) => file
        };
        let p = self.prime();
        let (max_mass_s, max_mass_t) = max_massey.into();
        // first identify kernels of left multiplication in this range
        let mut kernels: DashMap<(Bidegree,Bidegree), DashMap<FpVector,Subspace>> = DashMap::new();
        for s1 in 1..max_mass_s {
            for t1 in s1 as i32..max_mass_t {
                let dim1 = match self.num_gens(s1, t1) {
                    Some(n) => n,
                    None => { continue; } // not computed. this shouldn't happen
                };
                if dim1 == 0 {
                    continue; // no nonzero vectors
                }
                for v1 in AllVectorsIterator::new_whole_space(p, dim1) {
                    if v1.is_zero() { // no need to consider the 0 vector
                        continue;
                    }
                    // might go out of bounds if max_mass_s, max_mass_t > 0.5 max_s, max_t
                    // TODO
                    for s2 in 1..max_mass_s {
                        for t2 in s2 as i32..max_mass_t {
                            let (s3, t3) = (s1+s2, t1+t2);
                            let dim2 = match self.num_gens(s2, t2) {
                                Some(n) => n,
                                None => { continue; } // not computed. this shouldn't happen
                            };
                            let _dim3 = match self.num_gens(s3, t3) {
                                Some(n) => n,
                                None => { continue; } // not computed. this shouldn't happen
                            };
                            if dim2 == 0 {
                                continue; // no nonzero vectors
                            }
                            let lmul_v1 = match self.left_multiplication_by((s1, t1).into(), &v1, (s2, t2).into()) {
                                Ok(m) => m,
                                Err(_) => {
                                    continue;
                                }
                            };
                            let (aug_start, mut lmul_v1_aug) = Matrix::augmented_from_vec(p, &lmul_v1.to_vec());
                            lmul_v1_aug.row_reduce();
                            let kernel_lmul_v1 = lmul_v1_aug.compute_kernel(aug_start);
                            if kernel_lmul_v1.dimension() == 0 {
                                // kernel trival
                                continue; // skip
                            }
                            let bidegree_pair = ((s1,t1).into(),(s2,t2).into());
                            match kernels.get_mut(&bidegree_pair) {
                                Some(hm) => {
                                    hm.insert(v1.clone(),kernel_lmul_v1.clone());
                                },
                                None => {
                                    let mut new_hm = DashMap::new();
                                    new_hm.insert(v1.clone(), kernel_lmul_v1.clone());
                                    kernels.insert(bidegree_pair, new_hm);
                                }
                            }
                            /*
                            for v2 in AllVectorsIterator::new(&kernel_lmul_v1) {
                                if v2.is_zero() {
                                    continue;
                                }
                               eprintln!("({},{},{})*({},{},{}) = ({},{},{})", s1, t1, v1, s2, t2, v2, s3, t3, format!("0_{}", dim3));
                            }
                            */
                        }
                    }
                }
            }
        }
        // stores interesting (no element zero, and lands in nontrivial degree for now) massey-productable triples
        let mut triples: Vec<(AdamsElement, AdamsElement, AdamsElement)> = Vec::new();
        for s1 in 1..max_mass_s {
            for t1 in s1 as i32..max_mass_t {
                let deg1 = (s1, t1).into();
                for s2 in 1..max_mass_s {
                    for t2 in s2 as i32..max_mass_t {
                        let deg2 = (s2, t2).into();
                        let bideg_pr_l = (deg1, deg2);
                        let hm_kers = match kernels.get(&bideg_pr_l) {
                            Some(hm_kers) => {
                                hm_kers
                            },
                            None => { continue; } // no interesting vectors/kernels in this bidegree pair
                        };
                        for (v1, ker_v1) in hm_kers.iter() {
                            for v2 in AllVectorsIterator::new(ker_v1) {
                                if v2.is_zero() {
                                    continue; // skip the zero vector
                                }
                                // now iterate over s3, t3
                                for s3 in 1..max_mass_s {
                                    for t3 in s3 as i32..max_mass_t {
                                        let deg3 = (s3, t3).into();
                                        let bideg_pr_r = (deg2, deg3);
                                        let final_bideg = (s1+s2+s3-1, t1+t2+t3).into();
                                        match self.num_gens_bidegree(final_bideg) {
                                            Some(n) => {
                                                // computed bidegree
                                                if n==0 { // but dimension 0, no interesting massey products
                                                    continue;
                                                }
                                            },
                                            None => {
                                                // uncomputed bidegree, skip
                                                continue;
                                            }
                                        }
                                        let hm_kers_2 = match kernels.get(&bideg_pr_r) {
                                            Some(hm_kers_2) => { hm_kers_2 },
                                            None => { continue; } // no interesting vectors/kernels in this
                                            // bidegree pair
                                        };
                                        let ker_v2 = match hm_kers_2.get(&v2) {
                                            Some(ker_v2) => { ker_v2 },
                                            None => { continue; } // v2 doesn't have an interesting kernel here
                                        };
                                        for v3 in AllVectorsIterator::new(ker_v2) {
                                            if v3.is_zero() {
                                                continue;
                                            }
                                            triples.push(((s1,t1,v1.clone()).into(), (s2,t2,v2.clone()).into(), (s3, t3, v3.clone()).into()));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for (ae1, ae2, ae3) in &triples {
            let (s1,t1,v1) = ae1.into();
            let (s2,t2,v2) = ae2.into();
            let (s3,t3,v3) = ae3.into();
            let (shift_s,shift_t) = (s1+s2-1, t1+t2);
            let shift_n = shift_t-shift_s as i32;
            let (tot_s, tot_t) = (shift_s+s3, shift_t+t3);
            let tot_n = tot_t-tot_s as i32;
            let target_dim = match self.num_gens(tot_s, tot_t) {
                Some(n) => n,
                None => { continue; }
            };

            let res_hom_2 = self.adams_elt_to_resoln_hom(ae2);
            res_hom_2.extend_through_stem(shift_s, shift_n);
            let res_hom_3 = self.adams_elt_to_resoln_hom(ae3);
            res_hom_3.extend_through_stem(tot_s, tot_n);

            // TODO
            let homotopy = ChainHomotopy::new(
                &*self.resolution,
                &*self.resolution,
                s2+s3,
                t2+t3,
                |source_s, source_t, idx, row| {
                    let mid_s = source_s - s3;

                    res_hom_3.get_map(source_s)
                        .compose(res_hom_2.get_map(mid_s))
                        .apply_to_basis_element(row.as_slice_mut(), 1, source_t, idx);
                }
            );

            homotopy.extend(tot_s, tot_t);
            let last = homotopy.homotopy(tot_s);
            let mut answer = vec![0; target_dim];
            let mut nonzero = false;
            for i in 0..target_dim {
                let output = last.output(tot_t, i);
                for (k, entry) in v1.iter().enumerate() {
                    if entry != 0 {
                        answer[i] += entry * output.entry(k); // TODO: might need an offset here

                    }
                }
                if answer[i]!=0 {
                    nonzero=true;
                }
            }
            //for

            if nonzero {
                let massey_rep = FpVector::from_slice(v1.prime(), &answer);
                let indet = match self.compute_indeterminacy_of_massey_product(ae1, (s2, t2).into(), ae3) {
                    Ok((l_sub,r_sub)) => utils::subspace_sum(&l_sub, &r_sub),
                    Err(reason) => {
                       eprintln!("< ({}, {}, {}), ({}, {}, {}), ({}, {}, {}) > = ({}, {}, {}) + {:?}",
                            s1, t1, v1,
                            s2, t2, v2,
                            s3, t3, v3,
                            tot_s, tot_t, massey_rep,
                            format!("{} could not compute indeterminacy because {}", "{??}", reason)
                            );
                        // hopefully this doesn't happen
                        continue; // printed out, keep on going
                    }
                };
                print!("< ({}, {}, {}), ({}, {}, {}), ({}, {}, {}) > = ({}, {}, {}) + {:?}",
                    s1, t1, v1,
                    s2, t2, v2,
                    s3, t3, v3,
                    tot_s, tot_t, massey_rep,
                    indet
                );
                if indet.contains(massey_rep.as_slice()) {
                   eprintln!(" = 0 ")
                } else {
                   eprintln!();
                }
            } else {
                let _ = writeln!(zero_massey_output, "< ({}, {}, {}), ({}, {}, {}), ({}, {}, {}) > = 0 + did not compute indeterminacy",
                    s1, t1, v1,
                    s2, t2, v2,
                    s3, t3, v3,
                );
            }
        }
       eprintln!("{} total triples", triples.len());

    }
    */
}
