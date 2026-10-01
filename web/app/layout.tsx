/**
 * Root layout.
 *
 * Loads the one stylesheet and wraps every page in the shared chrome. There is
 * no client provider, no theme script, no font loader that reaches the network
 * at build time -- the type stack is system fonts declared in CSS -- because a
 * research instrument should render identically offline and in print, and a
 * static export has no server to lean on.
 */

import type { ReactNode } from "react";
import type { Metadata } from "next";

import "./globals.css";
import { Chrome } from "../components/nav";

export const metadata: Metadata = {
  title: "QDS — verifier's instrument",
  description:
    "Detection analytics for a teleportation-based quantum digital signature protocol: " +
    "measured error rates against the threshold ladder they are judged by.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Chrome>{children}</Chrome>
      </body>
    </html>
  );
}
