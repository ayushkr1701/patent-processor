import React from 'react'
import Main from './main'
import './App.css'

const App = () => {
  return (
    <div >
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
      }}>
        <img src="https://res.cloudinary.com/dtjivws2c/image/upload/v1718880280/diuty0bryxmiusio1cwo.png" width="50%" height="50%" alt="logo" />
      </div>
      <section className="neon-block">
        <div className="block">
          <span className="rainbow"></span>
          <Main />
        </div>
      </section>
    </div>
  )
}

export default App