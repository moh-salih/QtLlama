#pragma once
#include <llama.h>
#include <atomic>

#include <QtLlama/IEmbedder.h>
#include <QtLlama/Types.h>

namespace QtLlama {

class EmbeddingWorker : public IEmbedder {
    Q_OBJECT
public:
    explicit EmbeddingWorker(QObject *parent = nullptr);
    ~EmbeddingWorker() override;

public slots:
    void loadModel() override;
    void unloadModel() override;
    void generateEmbedding(const QString &text, int chunkIndex) override;
    void setConfig(QSharedPointer<EmbedConfig> config) override;
    void stop() override;
    void reset() override;

private:
    llama_model               * m_model   = nullptr;
    llama_context             * m_ctx     = nullptr;

    std::atomic<bool>           m_abort{false};
    QSharedPointer<EmbedConfig>   mConfig;
};

} // namespace QtLlama
