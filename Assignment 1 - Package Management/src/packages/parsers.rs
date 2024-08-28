use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use regex::Regex;

use crate::Packages;
use crate::packages::{Dependency, RelVersionedPackageNum};

use rpkg::debversion::{self, DebianVersionNum};

const KEYVAL_REGEX : &str = r"(?P<key>(\w|-)+): (?P<value>.+)";
const PKGNAME_AND_VERSION_REGEX : &str = r"(?P<pkg>(\w|\.|\+|-)+)( \((?P<op>(<|=|>)(<|=|>)?) (?P<ver>.*)\))?";

impl Packages {
    /// Loads packages and version numbers from a file, calling get_package_num_inserting on the package name
    /// and inserting the appropriate value into the installed_debvers map with the parsed version number.
    pub fn parse_installed(&mut self, filename: &str) {
        let kv_regexp = Regex::new(KEYVAL_REGEX).unwrap();
        if let Ok(lines) = read_lines(filename) {
            let mut current_package_num = 0;
            
            for line in lines {
                if let Ok(ip) = line {
                    // do something with ip
                    match kv_regexp.captures(&ip) {
                        None => (),
                        Some(caps) => {
                            let (key, value) = (caps.name("key").unwrap().as_str(),
                                                caps.name("value").unwrap().as_str());

                            if key == "Package" { 
                                current_package_num = self.get_package_num_inserting(&value);
                            }

                            if key == "Version" {
                                let debvers = value.trim().parse::<debversion::DebianVersionNum>().unwrap();
                                self.installed_debvers.insert(current_package_num, debvers);
                            } 
                            
                        }
                    }
                }
            }
        }
        println!("Packages installed: {}", self.installed_debvers.keys().len());
    }

    /// Loads packages, version numbers, dependencies, and md5sums from a file, calling get_package_num_inserting on the package name
    /// and inserting the appropriate values into the dependencies, md5sum, and available_debvers maps.
    pub fn parse_packages(&mut self, filename: &str) {
        let kv_regexp = Regex::new(KEYVAL_REGEX).unwrap();
        let pkgver_regexp = Regex::new(PKGNAME_AND_VERSION_REGEX).unwrap();

        if let Ok(lines) = read_lines(filename) {
            let mut current_package_num = 0;
            for line in lines {
                if let Ok(ip) = line {
                    // do more things with ip
                    match kv_regexp.captures(&ip) {
                        None => (),
                        Some(caps) => {
                            let (key, value) = (caps.name("key").unwrap().as_str(),
                                                caps.name("value").unwrap().as_str());

                            if key == "Package" { 
                                current_package_num = self.get_package_num_inserting(&value);
                            }

                            if key == "Version" {
                                let debvers = value.trim().parse::<debversion::DebianVersionNum>().unwrap();
                                self.available_debvers.insert(current_package_num, debvers);
                            }

                            if key == "MD5sum" {
                                let md5sum = value.to_string();
                                self.md5sums.insert(current_package_num, md5sum);
                            }
                            
                            if key == "Depends" {
                                let all_dependencies: Vec<&str> = value.split(",").collect();
                                let mut vec_dep: Vec<Dependency> = Vec::new();
                                for dependency in all_dependencies.iter() {
                                    let alternatives: Vec<&str> = dependency.split("|").collect();
                                    let mut vec_rel: Vec<RelVersionedPackageNum> = Vec::new();
                                    for alternative in alternatives.iter() {
                                        let mut rvpkgn = RelVersionedPackageNum {
                                            package_num : 0,
                                            rel_version: None,
                                        };
                                        match pkgver_regexp.captures(alternative)
                                        {
                                            None => (),
                                            Some(caps) => {                                           
                                                let (pkg, op, ver) = (caps.name("pkg").unwrap().as_str(),
                                                                    caps.name("op"),
                                                                    caps.name("ver"));
                                                if op.is_some() && ver.is_some() {
                                                    let parsed_op = op.unwrap().as_str().parse::<debversion::VersionRelation>().unwrap();
                                                    let parsed_ver = ver.unwrap().as_str().to_string();

                                                    rvpkgn = RelVersionedPackageNum {
                                                        package_num : self.get_package_num_inserting(&pkg),
                                                        rel_version: Some((parsed_op, parsed_ver)),
                                                    };
                                                }
                                                else {
                                                    rvpkgn = RelVersionedPackageNum {
                                                        package_num : self.get_package_num_inserting(&pkg),
                                                        rel_version: None,
                                                    };
                                                } 
                                            }
                                        }
                                        vec_rel.push(rvpkgn);
                                    }
                                    vec_dep.push(vec_rel);
                                }
                                self.dependencies.insert(current_package_num, vec_dep);
                            }
                        }
                    }
                }
            }
        }
        println!("Packages available: {}", self.available_debvers.keys().len()); 
    }
}


// standard template code downloaded from the Internet somewhere
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
