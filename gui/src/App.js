import React, { useState } from 'react';
import { motion } from 'framer-motion';

const styles = ['Classical', 'Jazz', 'Rock'];

function App() {
  const [model, setModel] = useState('');
  const [tempo, setTempo] = useState(120);
  const [style, setStyle] = useState(styles[0]);
  const [density, setDensity] = useState(0.5);

  const handlePlay = () => {
    // TODO: send params via WebSocket
    console.log({ model, tempo, style, density });
  };

  return (
    <div className="App">
      <h1>Mod Composer GUI</h1>
      <div>
        <label>Model</label>
        <select value={model} onChange={e => setModel(e.target.value)}>
          <option value="">Select model</option>
          <option value="base">Base</option>
          <option value="jazz">Jazz</option>
        </select>
      </div>
      <div>
        <label>Tempo: {tempo}</label>
        <input type="range" min="40" max="240" value={tempo} onChange={e => setTempo(e.target.value)} />
      </div>
      <div>
        <label>Style</label>
        <select value={style} onChange={e => setStyle(e.target.value)}>
          {styles.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>
      <div>
        <label>Density: {density}</label>
        <input type="range" min="0" max="1" step="0.01" value={density} onChange={e => setDensity(e.target.value)} />
      </div>
      <motion.button whileTap={{ scale: 0.9 }} onClick={handlePlay}>
        Play
      </motion.button>
    </div>
  );
}

export default App;
