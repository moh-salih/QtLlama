#pragma once
#include <QObject>
#include <QString>
#include <QList>
#include <QMetaType>
#include <vector>
#include <functional>



namespace QtLlama {
    Q_NAMESPACE

    enum class Status {
        Idle,
        Loading,
        Ready,
        Error
    };
    Q_ENUM_NS(Status);

    enum class Role {
        System,
        User,
        Assistant
    };
    Q_ENUM_NS(Role);

    struct Message {
        Role    role;
        QString content;
    };

    struct Config {
        QString         modelPath;
        int             threadCount             = 1;
        int             batchThreads            = 2;
        int             contextLength           = 0;        // 0 for auto-detecting context size from model.
        int             batchSize               = 2048;
        float           temperature             = 0.7f;
        float           topP                    = 0.9f;
        int             topK                    = 40;
        int             maxTokens               = -1;
        int             nGpuLayers              = 0;
        float           repeatPenalty           = 1.1f;
        int             repeatPenaltyLastN      = 64;
        float           penaltyFreq             = 0.0f;
        float           penaltyPresent          = 0.0f;
    };

    struct EmbedConfig {
        QString         modelPath;
        int             nCtx                    = 2048;
        int             nThreads                = 1;
        int             nGpuLayers              = 0;
    };


   

} // namespace QtLlama


Q_DECLARE_METATYPE(std::vector<float>)
Q_DECLARE_METATYPE(QtLlama::Message)
Q_DECLARE_METATYPE(QList<QtLlama::Message>)

