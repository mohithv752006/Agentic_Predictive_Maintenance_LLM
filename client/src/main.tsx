import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, createHashRouter } from "react-router";
import { RouterProvider } from "react-router/dom";
import { DEMO } from "./api/demo";
import { Layout } from "./components/Layout";
import "./index.css";
import { PAGES } from "./routes";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 15_000 } },
});

// Static hosting (GitHub Pages) has no server-side fallback, so the demo uses hash URLs.
const router = (DEMO ? createHashRouter : createBrowserRouter)([
  {
    path: "/",
    element: <Layout />,
    children: PAGES.map(({ path, component: Page }) => ({ path, element: <Page /> })),
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
);
