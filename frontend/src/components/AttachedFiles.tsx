type AttachedFilesProps = {
  files: File[]
  onRemove: (name: string) => void
}

export function AttachedFiles({ files, onRemove }: AttachedFilesProps) {
  if (!files.length) return null
  return (
    <ul className="mt-3 flex flex-wrap justify-center gap-2">
      {files.map((file) => (
        <li
          key={`${file.name}-${file.lastModified}`}
          className="flex items-center gap-2 rounded-full border border-white/20 bg-white/10 px-3 py-1 text-sm text-cream backdrop-blur-md"
        >
          <span className="max-w-[14rem] truncate">{file.name}</span>
          <button
            type="button"
            onClick={() => onRemove(file.name)}
            className="rounded-full px-1 text-cream/70 transition hover:text-cream focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold"
            aria-label={`Remove ${file.name}`}
          >
            ×
          </button>
        </li>
      ))}
    </ul>
  )
}
