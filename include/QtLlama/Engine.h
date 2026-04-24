#pragma once
#include <llama.h>
#include <atomic>
#include <QtLlama/IEngine.h>
#include <QtLlama/Types.h>
#include <functional>

namespace QtLlama {
    using Formatter = std::function<QString(const QList<Message>&)>;

class Engine : public IEngine {
    Q_OBJECT
public:
    explicit Engine(QObject *parent = nullptr);
    ~Engine() override;

public slots:
    void setConfig(QSharedPointer<Config> config) override;
    void loadModel() override;
    void unloadModel() override;
    void reloadModel() override;
    void generate(const QList<Message>& messages) override;
    void stop() override;
    void reset() override;

private:
    QString applyPromptFormat(const QList<Message>& messages) const;

    llama_model               * m_model   = nullptr;
    llama_context             * m_ctx     = nullptr;
    llama_sampler             * m_sampler = nullptr;
    Formatter                   mPromptFormatter;

    QSharedPointer<Config>      mConfig;
    std::atomic<bool>           m_abort{false};




    bool requiresReload(const Config& next, const Config& current);
    void buildSampler();
};

} // namespace QtLlama
