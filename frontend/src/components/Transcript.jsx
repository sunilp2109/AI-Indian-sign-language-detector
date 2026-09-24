export default function Transcript({ items = [] }) {
  return (
    <section className="panel history-panel" aria-labelledby="history-heading">
      <div className="panel-heading">
        <h2 id="history-heading">Translation history</h2>
        {items.length > 0 ? (
          <p className="panel-meta">{items.length} finalized</p>
        ) : null}
      </div>
      {items.length === 0 ? (
        <p className="hint">
          Finalized sentences appear here. Pause after signing, hold a complete
          phrase, or press Finalize. Nothing is spoken until you click Speak.
        </p>
      ) : (
        <ol className="history">
          {items.map((item) => (
            <li key={item.id || `${item.time}-${item.text}`}>
              <div className="history-meta">
                <time dateTime={item.time}>{item.time}</time>
                {item.reason ? <span className="history-reason">{item.reason}</span> : null}
              </div>
              <p className="history-text">{item.text}</p>
              {item.glosses?.length ? (
                <p className="history-glosses">{item.glosses.join(" → ")}</p>
              ) : null}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
