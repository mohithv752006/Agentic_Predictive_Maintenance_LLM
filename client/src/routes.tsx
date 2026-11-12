import type { ComponentType } from "react";
import DriveAnalysis from "./pages/DriveAnalysis";
import Escalations from "./pages/Escalations";
import Experiments from "./pages/Experiments";
import FleetOverview from "./pages/FleetOverview";
import Settings from "./pages/Settings";

/** Page registry: add a page = one file in pages/ + one entry here. */
export interface PageDef {
  path: string;
  label: string;
  component: ComponentType;
  nav: boolean;
  navPath?: string;
}

export const PAGES: PageDef[] = [
  { path: "/", label: "Fleet overview", component: FleetOverview, nav: true },
  { path: "/drives/:serial?", label: "Drive analysis", component: DriveAnalysis, nav: true, navPath: "/drives" },
  { path: "/experiments", label: "Experiments", component: Experiments, nav: true },
  { path: "/escalations", label: "Escalations", component: Escalations, nav: true },
  { path: "/settings", label: "Settings", component: Settings, nav: true },
];
