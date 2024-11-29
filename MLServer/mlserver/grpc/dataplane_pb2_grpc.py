# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import dataplane_pb2 as dataplane__pb2


class GRPCInferenceServiceStub(object):
    """
    Inference Server GRPC endpoints.

    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ServerLive = channel.unary_unary(
            "/inference.GRPCInferenceService/ServerLive",
            request_serializer=dataplane__pb2.ServerLiveRequest.SerializeToString,
            response_deserializer=dataplane__pb2.ServerLiveResponse.FromString,
        )
        self.ServerReady = channel.unary_unary(
            "/inference.GRPCInferenceService/ServerReady",
            request_serializer=dataplane__pb2.ServerReadyRequest.SerializeToString,
            response_deserializer=dataplane__pb2.ServerReadyResponse.FromString,
        )
        self.ModelReady = channel.unary_unary(
            "/inference.GRPCInferenceService/ModelReady",
            request_serializer=dataplane__pb2.ModelReadyRequest.SerializeToString,
            response_deserializer=dataplane__pb2.ModelReadyResponse.FromString,
        )
        self.ServerMetadata = channel.unary_unary(
            "/inference.GRPCInferenceService/ServerMetadata",
            request_serializer=dataplane__pb2.ServerMetadataRequest.SerializeToString,
            response_deserializer=dataplane__pb2.ServerMetadataResponse.FromString,
        )
        self.ModelMetadata = channel.unary_unary(
            "/inference.GRPCInferenceService/ModelMetadata",
            request_serializer=dataplane__pb2.ModelMetadataRequest.SerializeToString,
            response_deserializer=dataplane__pb2.ModelMetadataResponse.FromString,
        )
        self.ModelInfer = channel.unary_unary(
            "/inference.GRPCInferenceService/ModelInfer",
            request_serializer=dataplane__pb2.ModelInferRequest.SerializeToString,
            response_deserializer=dataplane__pb2.ModelInferResponse.FromString,
        )
        self.RepositoryIndex = channel.unary_unary(
            "/inference.GRPCInferenceService/RepositoryIndex",
            request_serializer=dataplane__pb2.RepositoryIndexRequest.SerializeToString,
            response_deserializer=dataplane__pb2.RepositoryIndexResponse.FromString,
        )
        self.RepositoryModelLoad = channel.unary_unary(
            "/inference.GRPCInferenceService/RepositoryModelLoad",
            request_serializer=dataplane__pb2.RepositoryModelLoadRequest.SerializeToString,
            response_deserializer=dataplane__pb2.RepositoryModelLoadResponse.FromString,
        )
        self.RepositoryModelUnload = channel.unary_unary(
            "/inference.GRPCInferenceService/RepositoryModelUnload",
            request_serializer=dataplane__pb2.RepositoryModelUnloadRequest.SerializeToString,
            response_deserializer=dataplane__pb2.RepositoryModelUnloadResponse.FromString,
        )


class GRPCInferenceServiceServicer(object):
    """
    Inference Server GRPC endpoints.

    """

    def ServerLive(self, request, context):
        """Check liveness of the inference server."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def ServerReady(self, request, context):
        """Check readiness of the inference server."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def ModelReady(self, request, context):
        """Check readiness of a model in the inference server."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def ServerMetadata(self, request, context):
        """Get server metadata."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def ModelMetadata(self, request, context):
        """Get model metadata."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def ModelInfer(self, request, context):
        """Perform inference using a specific model."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def RepositoryIndex(self, request, context):
        """Get the index of model repository contents."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def RepositoryModelLoad(self, request, context):
        """Load or reload a model from a repository."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def RepositoryModelUnload(self, request, context):
        """Unload a model."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_GRPCInferenceServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "ServerLive": grpc.unary_unary_rpc_method_handler(
            servicer.ServerLive,
            request_deserializer=dataplane__pb2.ServerLiveRequest.FromString,
            response_serializer=dataplane__pb2.ServerLiveResponse.SerializeToString,
        ),
        "ServerReady": grpc.unary_unary_rpc_method_handler(
            servicer.ServerReady,
            request_deserializer=dataplane__pb2.ServerReadyRequest.FromString,
            response_serializer=dataplane__pb2.ServerReadyResponse.SerializeToString,
        ),
        "ModelReady": grpc.unary_unary_rpc_method_handler(
            servicer.ModelReady,
            request_deserializer=dataplane__pb2.ModelReadyRequest.FromString,
            response_serializer=dataplane__pb2.ModelReadyResponse.SerializeToString,
        ),
        "ServerMetadata": grpc.unary_unary_rpc_method_handler(
            servicer.ServerMetadata,
            request_deserializer=dataplane__pb2.ServerMetadataRequest.FromString,
            response_serializer=dataplane__pb2.ServerMetadataResponse.SerializeToString,
        ),
        "ModelMetadata": grpc.unary_unary_rpc_method_handler(
            servicer.ModelMetadata,
            request_deserializer=dataplane__pb2.ModelMetadataRequest.FromString,
            response_serializer=dataplane__pb2.ModelMetadataResponse.SerializeToString,
        ),
        "ModelInfer": grpc.unary_unary_rpc_method_handler(
            servicer.ModelInfer,
            request_deserializer=dataplane__pb2.ModelInferRequest.FromString,
            response_serializer=dataplane__pb2.ModelInferResponse.SerializeToString,
        ),
        "RepositoryIndex": grpc.unary_unary_rpc_method_handler(
            servicer.RepositoryIndex,
            request_deserializer=dataplane__pb2.RepositoryIndexRequest.FromString,
            response_serializer=dataplane__pb2.RepositoryIndexResponse.SerializeToString,
        ),
        "RepositoryModelLoad": grpc.unary_unary_rpc_method_handler(
            servicer.RepositoryModelLoad,
            request_deserializer=dataplane__pb2.RepositoryModelLoadRequest.FromString,
            response_serializer=dataplane__pb2.RepositoryModelLoadResponse.SerializeToString,
        ),
        "RepositoryModelUnload": grpc.unary_unary_rpc_method_handler(
            servicer.RepositoryModelUnload,
            request_deserializer=dataplane__pb2.RepositoryModelUnloadRequest.FromString,
            response_serializer=dataplane__pb2.RepositoryModelUnloadResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "inference.GRPCInferenceService", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class GRPCInferenceService(object):
    """
    Inference Server GRPC endpoints.

    """

    @staticmethod
    def ServerLive(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/inference.GRPCInferenceService/ServerLive",
            dataplane__pb2.ServerLiveRequest.SerializeToString,
            dataplane__pb2.ServerLiveResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def ServerReady(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/inference.GRPCInferenceService/ServerReady",
            dataplane__pb2.ServerReadyRequest.SerializeToString,
            dataplane__pb2.ServerReadyResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def ModelReady(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/inference.GRPCInferenceService/ModelReady",
            dataplane__pb2.ModelReadyRequest.SerializeToString,
            dataplane__pb2.ModelReadyResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def ServerMetadata(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/inference.GRPCInferenceService/ServerMetadata",
            dataplane__pb2.ServerMetadataRequest.SerializeToString,
            dataplane__pb2.ServerMetadataResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def ModelMetadata(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/inference.GRPCInferenceService/ModelMetadata",
            dataplane__pb2.ModelMetadataRequest.SerializeToString,
            dataplane__pb2.ModelMetadataResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def ModelInfer(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/inference.GRPCInferenceService/ModelInfer",
            dataplane__pb2.ModelInferRequest.SerializeToString,
            dataplane__pb2.ModelInferResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def RepositoryIndex(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/inference.GRPCInferenceService/RepositoryIndex",
            dataplane__pb2.RepositoryIndexRequest.SerializeToString,
            dataplane__pb2.RepositoryIndexResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def RepositoryModelLoad(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/inference.GRPCInferenceService/RepositoryModelLoad",
            dataplane__pb2.RepositoryModelLoadRequest.SerializeToString,
            dataplane__pb2.RepositoryModelLoadResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def RepositoryModelUnload(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/inference.GRPCInferenceService/RepositoryModelUnload",
            dataplane__pb2.RepositoryModelUnloadRequest.SerializeToString,
            dataplane__pb2.RepositoryModelUnloadResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )