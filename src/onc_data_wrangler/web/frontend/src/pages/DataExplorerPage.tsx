import { useState, useEffect, useCallback } from 'react'
import { dataApi } from '@/api/client'
import { useAppStore } from '@/stores/appStore'
import {
  File,
  Folder,
  FolderOpen,
  ChevronRight,
  ChevronDown,
  Loader2,
  Table2,
  BarChart3,
  ChevronLeft,
} from 'lucide-react'
import { BrowseDialog } from '@/components/data/BrowseDialog'

interface FileEntry {
  path: string
  name: string
  size_bytes: number
  type: string
  columns: { name: string; type: string }[]
  row_count: number | null
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function DataExplorerPage() {
  const [pathInput, setPathInput] = useState('')
  const [files, setFiles] = useState<FileEntry[]>([])
  const [selectedFile, setSelectedFile] = useState<FileEntry | null>(null)
  const [previewData, setPreviewData] = useState<{
    columns: string[]
    rows: unknown[][]
    total_rows: number
  } | null>(null)
  const [isLoadingFiles, setIsLoadingFiles] = useState(false)
  const [isLoadingPreview, setIsLoadingPreview] = useState(false)
  const [page, setPage] = useState(0)
  const [columnStats, setColumnStats] = useState<{
    column: string
    dtype: string
    non_null_count: number
    unique_count: number
    top_values?: { value: string; count: number }[]
    numeric_stats?: { min: number; max: number; mean: number; median: number }
    [key: string]: unknown
  } | null>(null)
  const [selectedColumn, setSelectedColumn] = useState<string | null>(null)
  const [showBrowse, setShowBrowse] = useState(false)
  const pageSize = 50

  const handleLoadFiles = useCallback(async () => {
    if (!pathInput.trim()) return
    setIsLoadingFiles(true)
    try {
      const paths = pathInput.split(',').map((p) => p.trim()).filter(Boolean)
      const data = await dataApi.files(paths)
      setFiles(data as FileEntry[])
    } catch (err) {
      console.error('Failed to load files:', err)
    } finally {
      setIsLoadingFiles(false)
    }
  }, [pathInput])

  const handleSelectFile = useCallback(
    async (file: FileEntry) => {
      setSelectedFile(file)
      setPage(0)
      setSelectedColumn(null)
      setColumnStats(null)
      setIsLoadingPreview(true)
      try {
        const data = await dataApi.preview(file.path, pageSize, 0)
        setPreviewData(data)
      } catch (err) {
        console.error('Failed to load preview:', err)
      } finally {
        setIsLoadingPreview(false)
      }
    },
    []
  )

  const handlePageChange = useCallback(
    async (newPage: number) => {
      if (!selectedFile) return
      setPage(newPage)
      setIsLoadingPreview(true)
      try {
        const data = await dataApi.preview(selectedFile.path, pageSize, newPage * pageSize)
        setPreviewData(data)
      } catch (err) {
        console.error('Failed to load page:', err)
      } finally {
        setIsLoadingPreview(false)
      }
    },
    [selectedFile]
  )

  const handleColumnClick = useCallback(
    async (column: string) => {
      if (!selectedFile) return
      setSelectedColumn(column)
      try {
        const stats = await dataApi.columnStats(selectedFile.path, column) as typeof columnStats
        setColumnStats(stats)
      } catch {
        setColumnStats(null)
      }
    },
    [selectedFile]
  )

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      {/* Path input */}
      <div className="p-4 border-b bg-white">
        <div className="flex gap-2">
          <input
            type="text"
            value={pathInput}
            onChange={(e) => setPathInput(e.target.value)}
            placeholder="Enter paths to explore (comma-separated)..."
            className="flex-1 border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            onKeyDown={(e) => e.key === 'Enter' && handleLoadFiles()}
          />
          <button
            onClick={() => setShowBrowse(true)}
            className="flex items-center gap-2 px-3 py-2 border rounded-lg text-sm text-gray-600 hover:bg-gray-50"
          >
            <FolderOpen size={16} />
            Browse
          </button>
          <button
            onClick={handleLoadFiles}
            disabled={isLoadingFiles || !pathInput.trim()}
            className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:bg-gray-300 transition-colors text-sm"
          >
            {isLoadingFiles ? <Loader2 size={16} className="animate-spin" /> : <Folder size={16} />}
            Load
          </button>
        </div>
      </div>
      <BrowseDialog
        open={showBrowse}
        onClose={() => setShowBrowse(false)}
        onSelect={(paths) => {
          const existing = pathInput.trim()
          const newPaths = paths.join(', ')
          setPathInput(existing ? `${existing}, ${newPaths}` : newPaths)
        }}
        multiple
        title="Select Data Sources"
      />

      <div className="flex-1 flex overflow-hidden">
        {/* File list */}
        <div className="w-72 border-r overflow-y-auto bg-gray-50 p-2">
          {files.length === 0 && (
            <p className="text-xs text-gray-400 text-center mt-8">Enter paths above to browse files</p>
          )}
          {files.map((file) => (
            <button
              key={file.path}
              onClick={() => handleSelectFile(file)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm flex items-center gap-2 mb-1 transition-colors ${
                selectedFile?.path === file.path
                  ? 'bg-primary-100 text-primary-700'
                  : 'hover:bg-gray-100 text-gray-700'
              }`}
            >
              <File size={14} className="flex-shrink-0" />
              <div className="min-w-0 flex-1">
                <div className="truncate font-medium text-xs">{file.name}</div>
                <div className="text-xs text-gray-400">
                  {file.type} &middot; {formatBytes(file.size_bytes)}
                  {file.row_count !== null && ` &middot; ${file.row_count.toLocaleString()} rows`}
                </div>
              </div>
            </button>
          ))}
        </div>

        {/* Content area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {!selectedFile && (
            <div className="flex-1 flex items-center justify-center text-gray-400 text-sm">
              <div className="text-center">
                <Table2 size={32} className="mx-auto mb-2 text-gray-300" />
                Select a file to preview its contents
              </div>
            </div>
          )}

          {selectedFile && (
            <>
              {/* File info header */}
              <div className="px-4 py-2 border-b bg-gray-50 flex items-center justify-between">
                <div>
                  <span className="text-sm font-medium">{selectedFile.name}</span>
                  <span className="text-xs text-gray-500 ml-2">
                    {selectedFile.columns.length} columns
                    {previewData && ` &middot; ${previewData.total_rows.toLocaleString()} rows`}
                  </span>
                </div>
              </div>

              {/* Data table */}
              <div className="flex-1 overflow-auto">
                {isLoadingPreview ? (
                  <div className="flex items-center justify-center h-full">
                    <Loader2 size={24} className="animate-spin text-gray-400" />
                  </div>
                ) : previewData ? (
                  <table className="w-full text-xs border-collapse">
                    <thead className="sticky top-0 bg-gray-100">
                      <tr>
                        {previewData.columns.map((col) => (
                          <th
                            key={col}
                            onClick={() => handleColumnClick(col)}
                            className={`border-b border-r px-2 py-1.5 text-left font-medium whitespace-nowrap cursor-pointer hover:bg-gray-200 ${
                              selectedColumn === col ? 'bg-primary-100 text-primary-700' : ''
                            }`}
                          >
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {previewData.rows.map((row, i) => (
                        <tr key={i} className="hover:bg-blue-50">
                          {row.map((cell, j) => (
                            <td
                              key={j}
                              className="border-b border-r px-2 py-1 whitespace-nowrap max-w-[200px] truncate"
                              title={String(cell ?? '')}
                            >
                              {cell === null ? (
                                <span className="text-gray-300 italic">null</span>
                              ) : (
                                String(cell)
                              )}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : null}
              </div>

              {/* Pagination */}
              {previewData && previewData.total_rows > pageSize && (
                <div className="flex items-center justify-between px-4 py-2 border-t bg-gray-50">
                  <button
                    onClick={() => handlePageChange(page - 1)}
                    disabled={page === 0}
                    className="flex items-center gap-1 text-xs text-gray-600 hover:text-gray-800 disabled:text-gray-300"
                  >
                    <ChevronLeft size={14} /> Previous
                  </button>
                  <span className="text-xs text-gray-500">
                    {page * pageSize + 1}-{Math.min((page + 1) * pageSize, previewData.total_rows)} of{' '}
                    {previewData.total_rows.toLocaleString()}
                  </span>
                  <button
                    onClick={() => handlePageChange(page + 1)}
                    disabled={(page + 1) * pageSize >= previewData.total_rows}
                    className="flex items-center gap-1 text-xs text-gray-600 hover:text-gray-800 disabled:text-gray-300"
                  >
                    Next <ChevronRight size={14} />
                  </button>
                </div>
              )}
            </>
          )}
        </div>

        {/* Column stats sidebar */}
        {columnStats && selectedColumn && (
          <div className="w-64 border-l overflow-y-auto bg-gray-50 p-4">
            <div className="flex items-center gap-2 mb-3">
              <BarChart3 size={14} className="text-gray-500" />
              <h3 className="text-sm font-semibold">{selectedColumn}</h3>
            </div>
            <div className="space-y-2 text-xs">
              {Object.entries(columnStats).map(([key, val]) => {
                if (key === 'column' || key === 'top_values' || key === 'numeric_stats') return null
                return (
                  <div key={key} className="flex justify-between">
                    <span className="text-gray-500">{key.replace(/_/g, ' ')}</span>
                    <span className="font-medium">{String(val)}</span>
                  </div>
                )
              })}
              {columnStats.top_values && (
                <div className="mt-3">
                  <h4 className="text-gray-500 font-medium mb-1">Top Values</h4>
                  {columnStats.top_values.map((tv, i) => (
                    <div key={i} className="flex justify-between py-0.5">
                      <span className="truncate max-w-[120px]">{tv.value}</span>
                      <span className="text-gray-400">{tv.count}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
