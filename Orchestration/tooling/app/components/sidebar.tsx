"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"

const navItems = [
  { href: "/", label: "Dashboard", icon: "LayoutDashboard" },
  { href: "/runs/new", label: "New Run", icon: "Play" },
  { href: "/experiments", label: "Experiments", icon: "History" },
]

export function Sidebar() {
  const pathname = usePathname()

  return (
    <aside className="w-56 border-r bg-[#fafafa] flex flex-col">
      <div className="p-4 border-b">
        <h1 className="text-sm font-bold tracking-tight">LBEADS Hub</h1>
      </div>
      <nav className="flex-1 p-2">
        {navItems.map((item) => {
          const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href)
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "block px-3 py-2 text-sm",
                active
                  ? "bg-white border font-medium text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {item.label}
            </Link>
          )
        })}
      </nav>
    </aside>
  )
}
