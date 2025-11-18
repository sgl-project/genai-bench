#!/bin/bash

# Script to mirror GenAI-Bench image from Oracle internal registry to OCIR
# Requires Oracle Corporate Network access and Docker login to both registries

# Default version if not specified
DEFAULT_VERSION="0.1.83"
VERSION=${1:-$DEFAULT_VERSION}

# Registry URLs
SOURCE_REGISTRY="odo-docker-signed-local.artifactory-builds.oci.oraclecorp.com"
TARGET_REGISTRY="phx.ocir.io/idqj093njucb"
IMAGE_NAME="genai-bench"

echo "Mirroring GenAI-Bench image version: $VERSION"

# Pull from internal registry
echo "Pulling from internal registry..."
docker pull $SOURCE_REGISTRY/$IMAGE_NAME:$VERSION

# Tag for OCIR
echo "Tagging for OCIR..."
docker tag $SOURCE_REGISTRY/$IMAGE_NAME:$VERSION $TARGET_REGISTRY/$IMAGE_NAME:$VERSION

# Push to OCIR
echo "Pushing to OCIR..."
docker push $TARGET_REGISTRY/$IMAGE_NAME:$VERSION

echo "Successfully mirrored image version $VERSION" 