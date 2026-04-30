#pragma once
#include <QObject>
#include <QThread>
#include <QString>
#include <QList>
#include <QSharedPointer>
#include <QtLlama/IEngine.h>
#include <QtLlama/Types.h>

namespace QtLlama {

class Session : public QObject {
    Q_OBJECT
public:
    explicit Session(QObject *parent = nullptr);
    ~Session() override;

    void initialize(IEngine* engine);
    void setConfig(const Config& config);

    bool isGenerating() const { return mIsGenerating; }
    QtLlama::Status status() const { return mStatus; }
    QString statusText() const;

    void generate(const QList<QtLlama::Message>& messages, int sessionId);
    void generate(const QString& userMessage, int sessionId);
    void generate(const QString& userMessage, const QString& systemPrompt, int sessionId);

    void loadModel();
    void unloadModel();
    void reloadModel();
    void stop();

signals:
    void textGenerated(const QString& chunk, int sessionId);
    void responseReady(const QString& fullText, int sessionId);
    void isGeneratingChanged(bool isGenerating);
    void statusChanged(QtLlama::Status status);
    void errorOccurred(QtLlama::Error error);
    void reloadRequired();

private:
    IEngine                   * mEngine = nullptr;
    QThread                   * mWorkerThread = nullptr;
    QtLlama::Status             mStatus = QtLlama::Status::Idle;
    QSharedPointer<Config>      mConfig = QSharedPointer<Config>::create();

    bool                        mIsGenerating = false;
    std::atomic<int>            mCurrentSessionId{-1};
};

} // namespace QtLlama
