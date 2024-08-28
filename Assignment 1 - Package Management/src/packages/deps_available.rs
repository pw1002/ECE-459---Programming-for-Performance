use rpkg::debversion;
use crate::Packages;
use crate::packages::Dependency;

impl Packages {
    /// Gets the dependencies of package_name, and prints out whether they are satisfied (and by which library/version) or not.
    pub fn deps_available(&self, package_name: &str) {
        if !self.package_exists(package_name) {
            println!("no such package {}", package_name);
            return;
        }
        println!("Package {}:", package_name);
        let pkg_num: i32 = *self.get_package_num(package_name);
        let dependencies = self.dependencies.get(&pkg_num).unwrap();
        for deps in dependencies.iter() {
            let dep = self.dep2str(deps);
            println!("- dependency {:?}", dep);

            match self.dep_is_satisfied(deps) {
                Some(satisfied_pkg) => println!("+ {} satisfied by installed version {}", satisfied_pkg, self.get_installed_debver(satisfied_pkg).unwrap()),
                None => println!("+ {} not satisfied", dep),
            }
        }
    }

    /// Returns Some(package) which satisfies dependency dd, or None if not satisfied.
    pub fn dep_is_satisfied(&self, dd:&Dependency) -> Option<&str> {
        // presumably you should loop on dd
        for alts in dd.iter() {
            let pkg_name = self.get_package_name(alts.package_num);
            if !pkg_name.is_empty() {
                let installed_pkg_ver = self.get_installed_debver(pkg_name);
                if installed_pkg_ver != None {
                    if let Some((op, ver)) = &alts.rel_version {
                        if installed_pkg_ver != None {
                            let valid_version = debversion::cmp_debversion_with_op(op, installed_pkg_ver.unwrap(), &ver.parse::<debversion::DebianVersionNum>().unwrap());
                            if valid_version {
                                return Some(pkg_name);
                            }
                        }
                    }
                    else {
                        return Some(pkg_name);
                    }
                }      
            }
        }
        return None;
    }

    /// Returns a Vec of packages which would satisfy dependency dd but for the version.
    /// Used by the how-to-install command, which calls compute_how_to_install().
    pub fn dep_satisfied_by_wrong_version(&self, dd:&Dependency) -> Vec<&str> {
        assert! (self.dep_is_satisfied(dd).is_none());
        let mut result = vec![];
        // another loop on dd
        for alts in dd.iter() {
            let pkg_name = self.get_package_name(alts.package_num);
            if !pkg_name.is_empty() {
                if let Some((op, ver)) = &alts.rel_version {
                    let installed_pkg_ver = self.get_installed_debver(pkg_name);
                    if installed_pkg_ver != None {
                        let valid_version = debversion::cmp_debversion_with_op(op, installed_pkg_ver.unwrap(), &ver.parse::<debversion::DebianVersionNum>().unwrap());
                        if !valid_version {
                            result.push(pkg_name);
                        }
                    }
                }
            }
        }
        return result;
    }
}

