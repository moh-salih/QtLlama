#include <QtLlama/EmbeddingWorker.h>
#include <QtLlama/LlamaBackend.h>
#include <QDebug>

namespace QtLlama {

EmbeddingWorker::EmbeddingWorker(QObject *parent) : IEmbedder(parent) {
    QtLlama::ensureBackendInit();
}

EmbeddingWorker::~EmbeddingWorker() {
    unloadModel();
}

bool EmbeddingWorker::requiresReload(const EmbedConfig& next, const EmbedConfig& current) const {
    return next.modelPath  != current.modelPath
        || next.nCtx       != current.nCtx
        || next.nThreads   != current.nThreads
        || next.nGpuLayers != current.nGpuLayers;
}

void EmbeddingWorker::setConfig(QSharedPointer<EmbedConfig> config) {
    const bool needsReload = mConfig && requiresReload(*config, *mConfig);
    mConfig = config;
    if (needsReload && m_ctx) {
        if (mConfig->autoReload)
            { unloadModel(); loadModel(); }
        else
            emit reloadRequired();
    }
}

void EmbeddingWorker::loadModel() {
    unloadModel();
    emit modelStatusChanged(Status::Loading);

    if (!mConfig || mConfig->modelPath.isEmpty()) {
        qCritical() << "QtLlama:" << errorToString(Error::ModelPathEmpty);
        emit errorOccurred(Error::ModelPathEmpty);
        emit modelStatusChanged(Status::Error);
        return;
    }

    qInfo() << "=== LLAMA EMBEDDING ENGINE LOADING ===";
    qInfo() << "  Model Path:"   << mConfig->modelPath;
    qInfo() << "  Context Size:" << mConfig->nCtx;
    qInfo() << "  Thread Count:" << mConfig->nThreads;
    qInfo() << "  GPU Layers:"   << mConfig->nGpuLayers;

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = mConfig->nGpuLayers;

    m_model = llama_model_load_from_file(mConfig->modelPath.toStdString().c_str(), mp);
    if (!m_model) {
        qCritical() << "QtLlama:" << errorToString(Error::ModelLoadFailed);
        emit errorOccurred(Error::ModelLoadFailed);
        emit modelStatusChanged(Status::Error);
        return;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx        = mConfig->nCtx;
    cp.n_threads    = mConfig->nThreads;
    cp.embeddings   = true;

    m_ctx = llama_init_from_model(m_model, cp);
    if (!m_ctx) {
        qCritical() << "QtLlama:" << errorToString(Error::ContextInitFailed);
        emit errorOccurred(Error::ContextInitFailed);
        emit modelStatusChanged(Status::Error);
        return;
    }

    emit modelStatusChanged(Status::Ready);
}

void EmbeddingWorker::generateEmbedding(const QString &text, int chunkIndex) {
    if (!m_ctx || !m_model) return;

    qDebug() << "QtLlama: Generating embedding for chunk" << chunkIndex;
    emit isGeneratingChanged(true);
    m_abort.store(false);

    std::string stdText = text.toStdString();
    const llama_vocab *vocab = llama_model_get_vocab(m_model);

    std::vector<llama_token> tokens(stdText.size() + 32);
    int n_tokens = llama_tokenize(vocab, stdText.c_str(), stdText.size(),
                                  tokens.data(), tokens.size(), true, true);

    if (n_tokens < 0) {
        qCritical() << "QtLlama:" << errorToString(Error::TokenizationFailed);
        emit errorOccurred(Error::TokenizationFailed);
        emit isGeneratingChanged(false);
        return;
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);

    if (llama_decode(m_ctx, batch) != 0) {
        qCritical() << "QtLlama:" << errorToString(Error::DecodeFailed);
        emit errorOccurred(Error::DecodeFailed);
        emit isGeneratingChanged(false);
        return;
    }

    const int n_embd = llama_model_n_embd(m_model);
    const float *embd = llama_get_embeddings_seq(m_ctx, 0);

    if (embd) {
        std::vector<float> vector;
        vector.assign(embd, embd + n_embd);
        emit vectorReady(vector, text, chunkIndex);
    } else {
        qCritical() << "QtLlama:" << errorToString(Error::EmbeddingRetrieveFailed);
        emit errorOccurred(Error::EmbeddingRetrieveFailed);
    }

    emit isGeneratingChanged(false);
}

void EmbeddingWorker::unloadModel() {
    m_abort.store(true);
    if (m_ctx)   { llama_free(m_ctx);          m_ctx   = nullptr; }
    if (m_model) { llama_model_free(m_model);  m_model = nullptr; }
    emit modelStatusChanged(Status::Idle);
}

void EmbeddingWorker::reloadModel() {
    unloadModel();
    loadModel();
}

void EmbeddingWorker::stop()  { m_abort.store(true);  }
void EmbeddingWorker::reset() { m_abort.store(false); }

} // namespace QtLlama
