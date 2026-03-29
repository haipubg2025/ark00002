
import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ 
  content, 
  className = "", 
}) => {
  return (
    <div className={`markdown-body ${className}`}>
      <ReactMarkdown 
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={{
          p: ({ children }) => <p className="mb-4 leading-relaxed">{children}</p>,
          h1: ({ children }) => <h1 className="text-2xl font-bold mb-4 mt-6 text-mystic-accent">{children}</h1>,
          h2: ({ children }) => <h2 className="text-xl font-bold mb-3 mt-5 text-mystic-accent/90">{children}</h2>,
          h3: ({ children }) => <h3 className="text-lg font-bold mb-2 mt-4 text-mystic-accent/80">{children}</h3>,
          ul: ({ children }) => <ul className="list-disc pl-6 mb-4 space-y-1">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal pl-6 mb-4 space-y-1">{children}</ol>,
          li: ({ children }) => <li className="text-sm">{children}</li>,
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-mystic-accent/30 pl-4 py-1 my-4 italic bg-stone-200/30 dark:bg-slate-800/30 rounded-r">
              {children}
            </blockquote>
          ),
          code: ({ children }) => (
            <code className="px-1.5 py-0.5 bg-stone-300/50 dark:bg-slate-700/50 rounded font-mono text-xs">
              {children}
            </code>
          ),
          pre: ({ children }) => (
            <pre className="p-4 bg-stone-900 text-stone-100 rounded-lg overflow-x-auto my-4 text-xs font-mono">
              {children}
            </pre>
          ),
          table: ({ children }) => (
            <div className="overflow-x-auto my-6">
              <table className="min-w-full border-collapse border border-stone-300 dark:border-slate-700">
                {children}
              </table>
            </div>
          ),
          th: ({ children }) => (
            <th className="border border-stone-300 dark:border-slate-700 px-4 py-2 bg-stone-200 dark:bg-slate-800 font-bold text-left text-xs uppercase tracking-wider">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="border border-stone-300 dark:border-slate-700 px-4 py-2 text-sm">
              {children}
            </td>
          ),
          hr: () => <hr className="my-8 border-t border-stone-300 dark:border-slate-700" />,
          strong: ({ children }) => <strong className="font-bold text-mystic-accent/80">{children}</strong>,
          em: ({ children }) => <em className="italic opacity-90">{children}</em>,
          a: ({ href, children }) => (
            <a 
              href={href} 
              target="_blank" 
              rel="noopener noreferrer" 
              className="text-mystic-accent hover:underline decoration-mystic-accent/30 underline-offset-2"
            >
              {children}
            </a>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;
