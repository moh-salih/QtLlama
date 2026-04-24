#pragma once
#include <QObject>
#include <QString>
#include <vector>
#include <QSharedPointer>

#include <QtLlama/Types.h>

namespace QtLlama {

class IEmbedder : public QObject {
    Q_OBJECT
public:
    explicit IEmbedder(QObject *parent = nullptr) : QObject(parent) {}
    virtual ~IEmbedder() = default;

public slots:
    virtual void loadModel() = 0;
    virtual void unloadModel() = 0;
    virtual void reloadModel() = 0;
    virtual void generateEmbedding(const QString &text, int chunkIndex) = 0;
    virtual void setConfig(QSharedPointer<EmbedConfig> config) = 0;
    virtual void stop() = 0;
    virtual void reset() = 0;

signals:
    void vectorReady(const std::vector<float> &embedding, const QString &text, int chunkIndex);
    void modelStatusChanged(QtLlama::Status status);
    void isGeneratingChanged(bool isProcessing);
    void errorOccurred(const QString &message);
    void reloadRequired();
};

} // namespace QtLlama
