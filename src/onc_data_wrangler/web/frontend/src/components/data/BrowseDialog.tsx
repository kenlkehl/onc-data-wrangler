import { useState, useEffect, useCallback } from 'react'
import { dataApi } from '@/api/client'
import {
  Folder,
  File,
  ChevronUp,
  Check,
  X,
  Loader2,
  Home,
  FolderPlus,
} from 'lucide-react'

interface BrowseEntry {
  name: string
  path: string
  is_dir: boolean
  size_bytes?: number
  ext?: string
}

interface BrowseDialogProps {
  open: boolean
  onClose: () => void
  onSelect: (paths: string[]) => void
  /** Allow selecting multiple files/folders */
  multiple?: boolean
  /** Only show directories (for selecting output dirs) */
  dirOnly?: boolean
  /** Initial path to browse */
  initialPath?: string
  /** Dialog title */
  title?: string
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function BrowseDialog({
  open,
  onClose,
  onSelect,
  multiple = false,
  dirOnly = false,
  initialPath,
  title = 'Browse Files',
}: BrowseDialogProps) {
  const [currentPath, setCurrentPath] = useState('')
  const [parentPath, setParentPath] = useState<string | null>(null)
  const [entries, setEntries] = useState<BrowseEntry[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isCreatingFolder, setIsCreatingFolder] = useState(false)
  const [newFolderName, setNewFolderName] = useState('')
  const [newFolderError, setNewFolderError] = useState<string | null>(null)

  const loadDirectory = useCallback(async (path?: string) => {
    setIsLoading(true)
    setError(null)
    try {
      const res = await dataApi.browse(path)
      setCurrentPath(res.current_path)
      setParentPath(res.parent)
      setEntries(res.entries)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    if (open) {
      setSelected(new Set())
      loadDirectory(initialPath || undefined)
    }
  }, [open, initialPath, loadDirectory])

  const handleNavigate = (path: string) => {
    setSelected(new Set())
    loadDirectory(path)
  }

  const handleToggle = (entry: BrowseEntry) => {
    if (dirOnly && !entry.is_dir) return

    const newSelected = new Set(selected)
    if (newSelected.has(entry.path)) {
      newSelected.delete(entry.path)
    } else {
      if (!multiple) newSelected.clear()
      newSelected.add(entry.path)
    }
    setSelected(newSelected)
  }

  const handleCreateFolder = async () => {
    const name = newFolderName.trim()
    if (!name) return
    setNewFolderError(null)
    try {
      await dataApi.mkdir(`${currentPath}/${name}`)
      setIsCreatingFolder(false)
      setNewFolderName('')
      loadDirectory(currentPath)
    } catch (err) {
      setNewFolderError((err as Error).message)
    }
  }

  const handleConfirm = () => {
    if (selected.size === 0 && dirOnly) {
      // If dirOnly and nothing selected, use current directory
      onSelect([currentPath])
    } else if (selected.size > 0) {
      onSelect(Array.from(selected))
    }
    onClose()
  }

  if (!open) return null

  const filteredEntries = dirOnly
    ? entries.filter((e) => e.is_dir)
    : entries

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-xl shadow-2xl w-[600px] max-h-[70vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b">
          <h3 className="text-sm font-semibold">{title}</h3>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X size={18} />
          </button>
        </div>

        {/* Path bar */}
        <div className="flex items-center gap-2 px-4 py-2 bg-gray-50 border-b text-xs">
          <button
            onClick={() => handleNavigate('')}
            className="text-gray-500 hover:text-gray-700"
            title="Home"
          >
            <Home size={14} />
          </button>
          {parentPath && (
            <button
              onClick={() => handleNavigate(parentPath)}
              className="text-gray-500 hover:text-gray-700"
              title="Up"
            >
              <ChevronUp size={14} />
            </button>
          )}
          <span className="text-gray-700 font-mono truncate flex-1">{currentPath}</span>
          <button
            onClick={() => {
              setIsCreatingFolder(true)
              setNewFolderName('')
              setNewFolderError(null)
            }}
            className="flex items-center gap-1 text-gray-500 hover:text-gray-700 ml-auto flex-shrink-0"
            title="New Folder"
          >
            <FolderPlus size={14} />
            <span>New Folder</span>
          </button>
        </div>

        {/* File list */}
        <div className="flex-1 overflow-y-auto min-h-[200px]">
          {isCreatingFolder && (
            <div className="flex items-center gap-3 px-4 py-2 border-b border-gray-100 bg-amber-50">
              <Folder size={16} className="text-amber-500 flex-shrink-0" />
              <input
                type="text"
                value={newFolderName}
                onChange={(e) => setNewFolderName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleCreateFolder()
                  if (e.key === 'Escape') setIsCreatingFolder(false)
                }}
                placeholder="Folder name..."
                autoFocus
                className="flex-1 border rounded px-2 py-1 text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
              <button
                onClick={handleCreateFolder}
                disabled={!newFolderName.trim()}
                className="text-xs px-2 py-1 bg-primary-600 text-white rounded hover:bg-primary-700 disabled:bg-gray-300"
              >
                Create
              </button>
              <button
                onClick={() => setIsCreatingFolder(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X size={14} />
              </button>
            </div>
          )}
          {newFolderError && (
            <div className="px-4 py-1.5 text-xs text-red-600 bg-red-50 border-b border-red-100">
              {newFolderError}
            </div>
          )}
          {isLoading && (
            <div className="flex items-center justify-center h-32">
              <Loader2 size={20} className="animate-spin text-gray-400" />
            </div>
          )}
          {error && (
            <div className="p-4 text-sm text-red-600">{error}</div>
          )}
          {!isLoading && !error && filteredEntries.length === 0 && (
            <div className="p-4 text-sm text-gray-400 text-center">
              {dirOnly ? 'No subdirectories' : 'No files found'}
            </div>
          )}
          {!isLoading &&
            !error &&
            filteredEntries.map((entry) => (
              <div
                key={entry.path}
                className={`flex items-center gap-3 px-4 py-2 text-sm cursor-pointer border-b border-gray-50 ${
                  selected.has(entry.path)
                    ? 'bg-primary-50 text-primary-700'
                    : 'hover:bg-gray-50'
                } ${dirOnly && !entry.is_dir ? 'opacity-40 cursor-default' : ''}`}
                onClick={() => handleToggle(entry)}
                onDoubleClick={() => {
                  if (entry.is_dir) handleNavigate(entry.path)
                }}
              >
                {/* Selection indicator */}
                <div className={`w-4 h-4 rounded border flex items-center justify-center flex-shrink-0 ${
                  selected.has(entry.path)
                    ? 'bg-primary-500 border-primary-500 text-white'
                    : 'border-gray-300'
                }`}>
                  {selected.has(entry.path) && <Check size={10} />}
                </div>

                {/* Icon */}
                {entry.is_dir ? (
                  <Folder size={16} className="text-amber-500 flex-shrink-0" />
                ) : (
                  <File size={16} className="text-gray-400 flex-shrink-0" />
                )}

                {/* Name */}
                <span className="flex-1 truncate">{entry.name}</span>

                {/* Meta */}
                {entry.is_dir ? (
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      handleNavigate(entry.path)
                    }}
                    className="text-xs text-gray-400 hover:text-primary-600"
                  >
                    Open
                  </button>
                ) : (
                  entry.size_bytes != null && (
                    <span className="text-xs text-gray-400">{formatBytes(entry.size_bytes)}</span>
                  )
                )}
              </div>
            ))}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-3 border-t bg-gray-50">
          <span className="text-xs text-gray-500">
            {selected.size > 0
              ? `${selected.size} selected`
              : dirOnly
              ? 'Select a directory or use current'
              : 'Click to select, double-click to open folders'}
          </span>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-3 py-1.5 text-sm text-gray-600 hover:text-gray-800 border rounded-lg"
            >
              Cancel
            </button>
            {dirOnly && selected.size === 0 && (
              <button
                onClick={handleConfirm}
                className="px-3 py-1.5 text-sm bg-primary-600 text-white rounded-lg hover:bg-primary-700"
              >
                Use Current Dir
              </button>
            )}
            {(selected.size > 0 || !dirOnly) && (
              <button
                onClick={handleConfirm}
                disabled={selected.size === 0}
                className="px-3 py-1.5 text-sm bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:bg-gray-300"
              >
                Select{selected.size > 0 ? ` (${selected.size})` : ''}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
