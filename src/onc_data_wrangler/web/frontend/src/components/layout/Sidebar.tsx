import { NavLink } from 'react-router-dom'
import {
  Wand2,
  Play,
  Settings,
  Database,
  MessageSquare,
} from 'lucide-react'

const navItems = [
  { to: '/ui/setup', label: 'Setup', icon: Wand2 },
  { to: '/ui/pipeline', label: 'Pipeline', icon: Play },
  { to: '/ui/config', label: 'Config', icon: Settings },
  { to: '/ui/data', label: 'Data', icon: Database },
  { to: '/ui/chat', label: 'Chat', icon: MessageSquare },
]

export function Sidebar() {
  return (
    <aside className="w-56 bg-gray-900 text-gray-100 flex flex-col min-h-screen">
      <div className="p-4 border-b border-gray-700">
        <h1 className="text-lg font-bold tracking-tight">Talk-to-Data</h1>
        <p className="text-xs text-gray-400 mt-0.5">Data Wrangler UI</p>
      </div>
      <nav className="flex-1 py-2">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-2.5 text-sm transition-colors ${
                isActive
                  ? 'bg-primary-600 text-white'
                  : 'text-gray-300 hover:bg-gray-800 hover:text-white'
              }`
            }
          >
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="p-4 border-t border-gray-700 text-xs text-gray-500">
        Onc-Data-Wrangler v0.1
      </div>
    </aside>
  )
}
