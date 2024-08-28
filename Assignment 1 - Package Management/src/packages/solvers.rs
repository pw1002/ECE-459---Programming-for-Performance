use itertools::Itertools;

use crate::Packages;
use crate::packages::Dependency;
use std::collections::HashSet;
use rpkg::debversion::{DebianVersionNum, VersionRelation};
use rpkg::debversion;

impl Packages {
    /// Computes a solution for the transitive dependencies of package_name; when there is a choice A | B | C, 
    /// chooses the first option A. Returns a Vec<i32> of package numbers.
    ///
    /// Note: does not consider which packages are installed.
    pub fn transitive_dep_solution(&self, package_name: &str) -> Vec<i32> {
        if !self.package_exists(package_name) {
            return vec![];
        }
        let deps : &Vec<Dependency> = &*self.dependencies.get(self.get_package_num(package_name)).unwrap();
        let mut dependency_set: Vec<i32> = vec![];

        // implement worklist
        let mut worklist: Vec<i32> = Vec::new();

        // initialize worklist
        for dep in deps.iter() {
            worklist.push(dep.get(0).unwrap().package_num);
        }

        // iterate over worklist
        while let Some(item) = worklist.pop() {
            // get new dependencies
            let additional_deps = self.dependencies.get(&item).unwrap();
            // push new dependencies into worklist
            for dep in additional_deps.iter() {
                if !dependency_set.contains(&item) {
                    worklist.push(dep.get(0).unwrap().package_num);
                }
            }
            // add pkg num to set
            if !dependency_set.contains(&item) {
                dependency_set.push(item);
            }
        }

        return dependency_set;
    }

    /// Computes a set of packages that need to be installed to satisfy package_name's deps given the current installed packages.
    /// When a dependency A | B | C is unsatisfied, there are two possible cases:
    ///   (1) there are no versions of A, B, or C installed; pick the alternative with the highest version number (yes, compare apples and oranges).
    ///   (2) at least one of A, B, or C is installed (say A, B), but with the wrong version; of the installed packages (A, B), pick the one with the highest version number.
    pub fn compute_how_to_install(&self, package_name: &str) -> Vec<i32> {
        if !self.package_exists(package_name) {
            return vec![];
        }
        let deps : &Vec<Dependency> = &*self.dependencies.get(self.get_package_num(package_name)).unwrap();
        let mut dependencies_to_install : HashSet<i32> = HashSet::new();

        // implement more sophisticated worklist
        let mut worklist: Vec<Vec<i32>> = Vec::new();

        // initialize worklist
        for dep in deps.iter() {
            if self.dep_is_satisfied(dep) == None {
                let mut alternatives: Vec<i32> = Vec::new();
                for alt in dep.iter() {
                    alternatives.push(alt.package_num);
                }
                worklist.push(alternatives);
            }
        }

        // iterate over worklist
        while let Some(alternatives) = worklist.pop() {
            // get transitive dependencies
            for alt in alternatives.iter() {
                let transitive_deps = self.dependencies.get(alt).unwrap();
                // push these dependencies into the worklist after filtering
                for dep in transitive_deps.iter() {
                    // check if this dep was not already installed
                    if self.dep_is_satisfied(dep) == None {
                        // if there was at least 1 Dependency in (dep: Vec<Dependency>)
                        let most_valid_pkg_num = self.get_most_valid_option(dep);
                        if most_valid_pkg_num != -1 {
                            if !dependencies_to_install.contains(&most_valid_pkg_num) {
                                //println!("worklist.push: {}", self.get_package_name(most_valid_pkg_num));
                                worklist.push(vec![most_valid_pkg_num]);
                            }   
                        }
                    }
                }
                //println!("hashlist.push: {}", self.get_package_name(*alt));
                dependencies_to_install.insert(*alt);
            }
        }
        return dependencies_to_install.into_iter().collect();
    }

    // given a Dependency or a Vec<RelVerisionedPackageNum>, return the option based off the condition
    pub fn get_most_valid_option(&self, dependency: &Dependency) -> i32 {
        let mut highest_ver_pkg_num: i32 = -1;

        if dependency.len() < 1 {
            println!("get_most_valid_option() - No dependencies");
            return -1;
        }
        if dependency.len() == 1 {
            highest_ver_pkg_num = dependency.get(0).unwrap().package_num;
        }
        // there are alternatives (more than 1 dependency to choose from)
        else {
            let installed_wrong_version = self.dep_satisfied_by_wrong_version(dependency);
            // if there is one dep that is installed but wrong version
            if installed_wrong_version.len() == 1 {
                highest_ver_pkg_num = *self.get_package_num(installed_wrong_version.get(0).unwrap());
            }
            // otherwise, just compare apples and oranges
            else {
                highest_ver_pkg_num = dependency.get(0).unwrap().package_num;
                let mut highest_ver_pkg_ver: DebianVersionNum = dependency.get(0).unwrap().rel_version.as_ref().unwrap().1.parse::<debversion::DebianVersionNum>().unwrap();
                for alt in dependency.iter() {
                    let alt_pkg_num = alt.package_num;
                    let alt_ver: DebianVersionNum;
                    if let Some((_, ver)) = &alt.rel_version {
                        alt_ver = ver.parse::<debversion::DebianVersionNum>().unwrap();
                    }
                    else {
                        alt_ver = self.get_available_debver(self.get_package_name(alt_pkg_num)).unwrap().clone();
                    }
                    if debversion::cmp_debversion_with_op(&VersionRelation::StrictlyGreater, &alt_ver, &highest_ver_pkg_ver) {
                        highest_ver_pkg_num = alt_pkg_num;
                        highest_ver_pkg_ver = alt_ver;
                    }
                }
            }
        }

        return highest_ver_pkg_num;
    }

}
