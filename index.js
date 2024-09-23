require('dotenv').config();
const express = require('express');
const { OpenAI } = require('openai');
const bodyParser = require('body-parser');
const cosineSimilarity = require('compute-cosine-similarity');

const app = express();
const port = 3000;

// OpenAI configuration
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

// Middleware to parse JSON requests
app.use(bodyParser.json());

// Sample dataset
const data = [
    "Information about product A.",
    "Details regarding service B.",
    "FAQs about product C."
];

// Track total token usage
let totalTokenUsage = 0;

// Function to generate embeddings for the dataset
async function generateDataEmbeddings(inputs) {
    const embeddings = [];
    const batchSize = 100;

    for (let i = 0; i < inputs.length; i += batchSize) {
        const batch = inputs.slice(i, i + batchSize);

        try {
            const response = await openai.embeddings.create({
                model: 'text-embedding-ada-002',
                input: batch,
            });

            // Extract and print token usage for the batch
            const tokenUsage = response.usage.total_tokens;
            console.log(`Token usage for batch ${i / batchSize + 1}: ${tokenUsage}`);
            totalTokenUsage += tokenUsage; // Add to total token usage
            
            embeddings.push(...response.data.map(item => item.embedding));
        } catch (error) {
            console.error(`Error creating embeddings for batch starting at index ${i}:`, error);
            throw error;
        }
    }

    console.log(`Total token usage after embedding generation: ${totalTokenUsage}`);
    return embeddings;
}

// Generate embeddings for the dataset at startup
let dataEmbeddings;
generateDataEmbeddings(data).then(embeddings => {
    dataEmbeddings = embeddings;
}).catch(err => {
    console.error('Error generating embeddings:', err);
});

// Route for semantic search
app.post('/search', async (req, res) => {
    const userQuery = req.body.query;

    try {
        const queryResponse = await openai.embeddings.create({
            model: 'text-embedding-ada-002',
            input: [userQuery],
        });
        const queryEmbedding = queryResponse.data[0].embedding;
        const searchTokenUsage = queryResponse.usage.total_tokens;
        console.log(`Token usage for search query: ${searchTokenUsage}`);

        const similarities = dataEmbeddings.map(
            embedding => cosineSimilarity(queryEmbedding, embedding));
        const bestMatchIndex = similarities.indexOf(Math.max(...similarities));

        res.json({
            searchTokenUsage: searchTokenUsage,
            bestMatch: data[bestMatchIndex],
            similarity: similarities[bestMatchIndex],
        });
    } catch (error) {
        console.error('Error processing search:', error);
        res.status(500).send('Error processing search');
    }
});

// Start the server
app.listen(port, () => {
    console.log(`Server is running at http://localhost:${port}`);
});
