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
    Q_PROPERTY(QtLlama::Status status READ status NOTIFY statusChanged)
    Q_PROPERTY(bool isGenerating READ isGenerating NOTIFY isGeneratingChanged)
    Q_PROPERTY(QString statusText READ statusText NOTIFY statusChanged)

public:
    explicit Embedder(QObject *parent = nullptr);
    ~Embedder() override;

    void initialize(IEmbedder * engine);
    void setConfig(const EmbedConfig& config);

    void loadModel();
    void unloadModel();
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

private:
    IEmbedder                   * mEngine = nullptr;
    QThread                     * mWorkerThread = nullptr;
    QSharedPointer<EmbedConfig>   mConfig = QSharedPointer<EmbedConfig>::create();
    QtLlama::Status               mStatus = QtLlama::Status::Idle;
    bool                          mIsGenerating = false;
};

} // namespace QtLlama
