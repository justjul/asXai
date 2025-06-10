# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.


## Here's now my memo on how I made it work:
0- intsall Node.js using instruction in https://learn.microsoft.com/en-us/windows/dev-environment/javascript/nodejs-on-wsl#install-nvm-nodejs-and-npm
1- cd src
2- npm create vite@latest frontend -- --template react
3- npm install
4- npm install -D tailwindcss postcss autoprefixer
5- npm install tailwindcss @tailwindcss/vite
6- npm install class-variance-authority clsx tailwind-variants
7- npm install lucide-react
8- npm install react-markdown remark-gfm
9- npm install react-router-dom
10- npm run firebase
11- npm install eventsource-parser
12- npm run dev

then start ngrok tunnel: ngrok start --all