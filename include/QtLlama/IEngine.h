#pragma once
#include <QObject>
#include <QString>
#include <QList>
#include <QSharedPointer>

#include <QtLlama/Types.h>

namespace QtLlama {

class IEngine : public QObject {
    Q_OBJECT
public:
    explicit IEngine(QObject *parent = nullptr) : QObject(parent) {}
    virtual ~IEngine() = default;

public slots:
    virtual void setConfig(QSharedPointer<Config> config) = 0;
    virtual void loadModel() = 0;
    virtual void unloadModel() = 0;
    virtual void reloadModel() = 0;
    virtual void generate(const QList<Message>& messages) = 0;
    virtual void stop() = 0;
    virtual void reset() = 0;

signals:
    void tokenGenerated(const QString &textChunk);
    void responseReady(const QString& fullText);
    void modelStatusChanged(QtLlama::Status status);
    void isGeneratingChanged(bool isGenerating);
    void errorOccurred(const QString &message);
    void reloadRequired();
};

} // namespace QtLlama
