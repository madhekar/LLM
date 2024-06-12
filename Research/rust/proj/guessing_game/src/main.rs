use std::io;

fn main() {
  println!("input the guess number!");

  let mut guess = String::new();
  
  io::stdin()
     .read_line(&mut guess)
     .expect("failed to readline"); 
println!("number guess: {guess}");	
}
