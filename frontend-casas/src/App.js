import React, { useState } from 'react';
import './App.css';

function App() {
  // Estados para guardar o que o humano digita
  const [area, setArea] = useState('');
  const [quartos, setQuartos] = useState('');
  const [preco, setPreco] = useState(null);
  const [carregando, setCarregando] = useState(false);

  // Função que chama o Oráculo (API Flask)
  const fazerPrevisao = async (e) => {
    e.preventDefault(); // Não recarregar a página
    setCarregando(true);

    try {
      const resposta = await fetch('http://127.0.0.1:5000/prever', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          area: area,
          quartos: quartos
        }),
      });

      const dados = await resposta.json();
      setPreco(dados.preco_previsto);
    } catch (erro) {
      console.error("Erro ao consultar o oráculo:", erro);
      alert("O Oráculo não respondeu. Verifique se o arquivo 'app.py' está rodando!");
    }
    setCarregando(false);
  };

  return (
    <div className="App" style={{ padding: '50px', fontFamily: 'Arial' }}>
      <h1>🏠 Oráculo Imobiliário</h1>
      <p>Digite os dados da casa para prever o valor divino.</p>

      <form onSubmit={fazerPrevisao} style={{ maxWidth: '400px', margin: '0 auto' }}>
        
        <div style={{ marginBottom: '20px' }}>
          <label>Área (m²):</label><br/>
          <input 
            type="number" 
            value={area}
            onChange={(e) => setArea(e.target.value)}
            required
            style={{ padding: '10px', width: '100%' }}
          />
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label>Quantidade de Quartos:</label><br/>
          <input 
            type="number" 
            value={quartos}
            onChange={(e) => setQuartos(e.target.value)}
            required
            style={{ padding: '10px', width: '100%' }}
          />
        </div>

        <button 
          type="submit" 
          style={{ 
            padding: '15px 30px', 
            backgroundColor: '#282c34', 
            color: 'white', 
            border: 'none', 
            cursor: 'pointer',
            fontSize: '16px'
          }}
        >
          {carregando ? 'Consultando os Astros...' : 'Prever Preço'}
        </button>
      </form>

      {preco && (
        <div style={{ marginTop: '40px', padding: '20px', border: '2px solid #4caf50', borderRadius: '10px' }}>
          <h2>💰 Valor Estimado:</h2>
          <h1 style={{ color: '#4caf50' }}>R$ {preco}</h1>
        </div>
      )}
    </div>
  );
}

export default App;