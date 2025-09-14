import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db/drizzle';
import { listings, images, listingsEnglish } from '@/lib/db/schema';
import { sql, eq, and, inArray } from 'drizzle-orm';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ listingId: string }> }
) {
  console.log('🔍 [SimilarAPI] API route hit!');
  
  try {
    const { listingId } = await params;
    console.log('🔍 [SimilarAPI] Listing ID:', listingId);

    if (!listingId) {
      return NextResponse.json({ error: 'Listing ID is required' }, { status: 400 });
    }

    // First check if this listing has a vector in the hokkaido_lfm2_fused_mlp table
    console.log('🔍 [SimilarAPI] Looking for vector for listing ID:', listingId, 'type:', typeof listingId);
    
    const vectorResult = await db.execute(
      sql`SELECT id, vec FROM vecs.hokkaido_lfm2_fused_mlp WHERE id = ${listingId}`
    );
    
    console.log('🔍 [SimilarAPI] Vector result length:', vectorResult.length || 0);

    if (!vectorResult || vectorResult.length === 0) {
      // If no vector exists for this listing, return empty results
      return NextResponse.json({ 
        listings: [],
        message: 'No vector embedding available for this listing'
      }, { status: 200 });
    }

    const vectorData = vectorResult[0] as any;

    // Get the current listing's details for hybrid filtering
    const currentListingResult = await db
      .select({
        lat: listings.lat,
        lng: listings.lng,
        price: listings.price
      })
      .from(listings)
      .where(eq(listings.listingId, listingId))
      .limit(1);

    if (currentListingResult.length === 0) {
      return NextResponse.json({ 
        listings: [],
        message: 'Current listing not found'
      }, { status: 404 });
    }

    const currentListing = currentListingResult[0];
    
    // Check if location data exists
    if (!currentListing.lat || !currentListing.lng || !currentListing.price) {
      return NextResponse.json({ 
        listings: [],
        message: 'Current listing missing required data (location or price)'
      }, { status: 400 });
    }
    
    const priceRange = {
      min: currentListing.price * 0.5, // -50% (more flexible)
      max: currentListing.price * 2.0  // +100% (more flexible)
    };
    const locationRange = {
      latMin: currentListing.lat - 0.2, // ~5km (wider area)
      latMax: currentListing.lat + 0.2,
      lngMin: currentListing.lng - 0.2,
      lngMax: currentListing.lng + 0.2
    };

    // Debug logging
    console.log('🔍 [HybridSearch] Current listing:', {
      id: listingId,
      lat: currentListing.lat,
      lng: currentListing.lng,
      price: currentListing.price
    });
    console.log('🔍 [HybridSearch] Search ranges:', { priceRange, locationRange });

    // Step 1: Apply location and price filters FIRST (hard filters)
    const locationPriceFiltered = await db
      .select({
        listingId: listings.listingId,
        lat: listings.lat,
        lng: listings.lng,
        price: listings.price
      })
      .from(listings)
      .where(
        and(
          sql`${listings.lat} BETWEEN ${locationRange.latMin} AND ${locationRange.latMax}`,
          sql`${listings.lng} BETWEEN ${locationRange.lngMin} AND ${locationRange.lngMax}`,
          sql`${listings.price} BETWEEN ${priceRange.min} AND ${priceRange.max}`,
          eq(listings.isActive, true),
          sql`${listings.listingId} != ${listingId}`
        )
      );

    console.log('🔍 [HybridSearch] Location+Price filtered candidates:', locationPriceFiltered.length);

    if (locationPriceFiltered.length === 0) {
      console.log('🔍 [HybridSearch] No candidates after location+price filtering');
    }

    // Step 2: For the location+price filtered results, check which have vectors and rank by similarity
    const candidateIds = locationPriceFiltered.map(l => l.listingId);
    let similarResult = null;

    if (candidateIds.length > 0) {
      // Use the original function but filter the results to only include our candidates
      const allVectorResults = await db.execute(
        sql`SELECT * FROM match_listings_hokkaido_lfm2(${vectorData.vec}::vector, 0.5, 100)`
      );
      
      // Filter to only candidates that passed location+price filters
      const candidateSet = new Set(candidateIds);
      similarResult = (allVectorResults as any[])
        .filter((result: any) => candidateSet.has(result.id) && result.id !== listingId)
        .slice(0, 6);
        
      console.log('🔍 [HybridSearch] Filtered vector results to candidates:', similarResult.length);
    }

    console.log('🔍 [HybridSearch] Initial results:', similarResult?.length || 0);

    // Only use results that passed location+price filters - NO fallback to unrestricted vector search
    const finalResult = similarResult;

    if (!finalResult || finalResult.length === 0) {
      return NextResponse.json({ 
        listings: [],
        message: 'No similar listings found'
      }, { status: 200 });
    }

    // Extract listing IDs from results
    const similarListingIds = (finalResult as any[])
      .slice(0, 5) // Take only 5 similar listings
      .map((v: any) => v.id);

    if (similarListingIds.length === 0) {
      return NextResponse.json({ 
        listings: [],
        message: 'No similar listings found'
      }, { status: 200 });
    }

    // Fetch full listing details for similar listings using Drizzle
    const similarListings = await db
      .select({
        listingId: listings.listingId,
        title: listings.title,
        price: listings.price,
        previousPrice: listings.previousPrice,
        location: listings.location,
        sizeSqm: listings.sizeSqm,
        rooms: listings.rooms,
        yearBuilt: listings.yearBuilt,
        listingType: listings.listingType,
        listingTypeEn: listings.listingTypeEn,
        listingUrl: listings.listingUrl,
        firstSeen: listings.firstSeen,
        lastSeen: listings.lastSeen,
        lat: listings.lat,
        lng: listings.lng,
        titleEnglish: listingsEnglish.titleEnglish,
        locationEnglish: listingsEnglish.locationEnglish,
        imageId: images.id,
        imageSourceUrl: images.sourceUrl,
        imageIsMain: images.isMain,
        imageCloudfrontUrl: images.cloudfrontUrl,
        imageS3Key: images.s3Key,
        imageS3BucketName: images.s3BucketName,
      })
      .from(listings)
      .leftJoin(listingsEnglish, eq(listings.listingId, listingsEnglish.listingId))
      .leftJoin(images, eq(listings.listingId, images.listingId))
      .where(
        and(
          inArray(listings.listingId, similarListingIds),
          eq(listings.isActive, true)
        )
      )
      .orderBy(listings.price);

    // Transform the flat results into grouped listing objects with images
    const listingMap = new Map<string, any>();
    
    for (const row of similarListings) {
      if (!listingMap.has(row.listingId)) {
        listingMap.set(row.listingId, {
          listingId: row.listingId,
          title: row.title,
          titleEnglish: row.titleEnglish,
          price: parseFloat(row.price?.toString() || '0'),
          previousPrice: row.previousPrice ? parseFloat(row.previousPrice.toString()) : null,
          location: row.location,
          locationEnglish: row.locationEnglish,
          sizeSqm: parseFloat(row.sizeSqm?.toString() || '0'),
          rooms: row.rooms,
          yearBuilt: row.yearBuilt,
          listingType: row.listingType,
          listingTypeEn: row.listingTypeEn,
          listingUrl: row.listingUrl,
          firstSeen: row.firstSeen,
          lastSeen: row.lastSeen,
          lat: row.lat,
          lng: row.lng,
          images: []
        });
      }
      
      // Add image if it exists
      if (row.imageId) {
        listingMap.get(row.listingId).images.push({
          id: row.imageId,
          sourceUrl: row.imageSourceUrl,
          isMain: row.imageIsMain,
          cloudfrontUrl: row.imageCloudfrontUrl,
          s3Key: row.imageS3Key,
          s3BucketName: row.imageS3BucketName
        });
      }
    }
    
    const transformedListings = Array.from(listingMap.values());

    return NextResponse.json({ 
      listings: transformedListings,
      count: transformedListings.length
    });

  } catch (error) {
    console.error('Error in similar listings API:', error);
    return NextResponse.json({ 
      error: 'Internal server error',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}