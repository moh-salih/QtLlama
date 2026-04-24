#pragma once
#include <QObject>
#include <QThread>
#include <vector>
#include <QString>
#include <QSharedPointer>
#include <QtLlama/IEmbedder.h>
#include <QtLlama/Types.h>

namespace QtLlama {

class Embedder : public QObject {
    Q_OBJECT
public:
    explicit Embedder(QObject *parent = nullptr);
    ~Embedder() override;

    void initialize(IEmbedder * engine);
    void setConfig(const EmbedConfig& config);

    void loadModel();
    void unloadModel();
    void reloadModel();
    void generateEmbedding(const QString& text, int chunkIndex);
    void stop();

    QtLlama::Status status() const { return mStatus; }
    bool isGenerating() const { return mIsGenerating; }
    QString statusText() const;

signals:
    void embeddingReady(const std::vector<float>& embedding, const QString& text, int chunkIndex);
    void statusChanged(QtLlama::Status status);
    void isGeneratingChanged(bool);
    void errorOccurred(const QString& msg);
    void reloadRequired();

private:
    IEmbedder                   * mEngine = nullptr;
    QThread                     * mWorkerThread = nullptr;
    QSharedPointer<EmbedConfig>   mConfig = QSharedPointer<EmbedConfig>::create();
    QtLlama::Status               mStatus = QtLlama::Status::Idle;
    bool                          mIsGenerating = false;
};

} // namespace QtLlama
