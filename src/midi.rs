use std::sync::{Arc, RwLock};

use midir::{Ignore, MidiInput, MidiInputConnection};

pub type State = Arc<RwLock<(u32, [f32; 8])>>;

pub struct Midi {
    pub state: State,
    pub conn: MidiInputConnection<()>,
}

impl Midi {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let state = Arc::new(RwLock::new((0, [0.0; 8])));
        let mut midi_in = MidiInput::new("midir reading input")?;
        midi_in.ignore(Ignore::None);
        let in_ports = midi_in.ports();
        let in_port = in_ports
            .iter()
            .find(|port| midi_in.port_name(port).unwrap().starts_with("LPD8"))
            .expect("LPD8 not found!");

        println!("\nOpening connection");
        let in_port_name = midi_in.port_name(in_port)?;

        let conn = midi_in.connect(
            in_port,
            "strange-attractor-input",
            {
                let state = state.clone();
                move |stamp, message, _| {
                    if message[0] == 144 && message[1] >= 36 && message[1] <= 43 {
                        state.write().unwrap().0 = (message[1] - 36) as u32;
                    }
                    if message[0] == 176 && message[1] > 0 && message[1] < 9 {
                        state.write().unwrap().1[(message[1] - 1) as usize] =
                            message[2] as f32 / 127.0;
                    }
                    println!("{}: {:?} (len = {})", stamp, message, message.len());
                }
            },
            (),
        )?;

        println!(
            "Connection open, reading input from '{}' (press enter to exit) ...",
            in_port_name
        );

        Ok(Self { state, conn })
    }
}
